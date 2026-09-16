"""FP16 neural simulation with an FP32 surrogate-gradient backward pass on Metal.

All neurons, edges and 0.1 ms ticks are retained. A command window is one custom
VJP: the tiny adapter/Adam state, reductions and backward adjoints remain FP32.
"""
from functools import lru_cache
import math
import numpy as np
import torch


@lru_cache(maxsize=16)
def training_kernels(surrogate_scale=1.):
    if not math.isfinite(surrogate_scale) or not 0 < surrogate_scale <= 1:
        raise ValueError('Surrogate scale must be in (0, 1]')
    source = r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void integrate(device half* v, device half* g, device half* spikes,
    device long* last, const device long* refractory, const device half* drive,
    const device bool* silence, const device half* gain, device half* future,
    device half* raw, device int* flags, const device half* p, constant long& tick,
    uint i [[thread_position_in_grid]]) {
    bool free = tick-last[i] >= refractory[i];
    if (free) { v[i] = -52.h + (v[i]+52.h)*p[0] + g[i]*p[2] + drive[i]*p[3]; g[i] *= p[1]; }
    if (silence[i]) v[i] = -52.h;
    bool fired = free && v[i] > -45.h;
    raw[i] = v[i];
    flags[i] = int(free) | (int(fired)<<1) | (int(silence[i])<<2);
    spikes[i] = half(fired);
    if (fired) last[i] = tick;
    future[i] += half(fired)*gain[i];
}
kernel void finish(device half* v, device half* g, const device half* spikes,
    const device int* flags, const device half* incoming, device half* consumed,
    device half* emitted, device long* counts, device half* rates, const device half* p,
    uint i [[thread_position_in_grid]]) {
    bool fired = (flags[i]&2)!=0, receiving = (flags[i]&3)==1;
    g[i] += incoming[i]*half(receiving);
    if (fired) { v[i] = -52.h; g[i] = 0.h; }
    emitted[i] = consumed[i]; consumed[i] = 0.h;
    counts[i] += long(spikes[i]);
    rates[i] = rates[i]*p[4] + spikes[i]*p[5];
}
kernel void incoming_grad(const device float* dg, const device int* flags,
    device float* di, uint i [[thread_position_in_grid]]) {
    di[i] = (flags[i]&3)==1 ? dg[i] : 0.f;
}
kernel void transpose_grad(const device int* ptr, const device int* post,
    const device half* weight, const device float* di, device float* dq,
    uint tid [[thread_position_in_grid]]) {
    uint i=tid/32, lane=tid%32; float sum=0.f;
    for (int e=ptr[i]+int(lane); e<ptr[i+1]; e+=32) sum += float(weight[e])*di[post[e]];
    float reduced=simd_sum(sum); if (lane==0) dq[i]=reduced;
}
kernel void factor_grad(const device int* ptr, const device int* pre,
    const device int* post, const device float* base, const device half* emitted,
    const device float* di, device float* df, uint tid [[thread_position_in_grid]]) {
    uint i=tid/32, lane=tid%32; float sum=0.f;
    for (int e=ptr[i]+int(lane); e<ptr[i+1]; e+=32)
        sum += base[e]*float(emitted[pre[e]])*di[post[e]];
    float reduced=simd_sum(sum); if (lane==0) df[i]=reduced;
}
kernel void active_factor_grad(const device int* ptr, const device int* source,
    const device int* post, const device float* base, const device half* emitted,
    const device float* di, device float* df, uint tid [[thread_position_in_grid]]) {
    uint i=tid/32, lane=tid%32;
    float q=float(emitted[source[i]]), sum=0.f;
    if (q!=0.f) {
        for (int e=ptr[i]+int(lane); e<ptr[i+1]; e+=32)
            sum += base[e]*q*di[post[e]];
    }
    float reduced=simd_sum(sum); if (lane==0) df[i]=reduced;
}
kernel void reduce_factors(const device int* ptr, const device float* partial,
    device float* df, uint tid [[thread_position_in_grid]]) {
    uint i=tid/32, lane=tid%32; float sum=0.f;
    for (int e=ptr[i]+int(lane); e<ptr[i+1]; e+=32) sum+=partial[e];
    float reduced=simd_sum(sum); if (lane==0) df[i]+=reduced;
}
kernel void reverse_state(device float* dv, device float* dg, device float* dr,
    const device float* future_grad, const device half* gain, const device half* raw,
    const device int* flags, const device half* p, uint i [[thread_position_in_grid]]) {
    bool free=(flags[i]&1)!=0, fired=(flags[i]&2)!=0, silent=(flags[i]&4)!=0;
    float ds=dr[i]*float(p[5])+future_grad[i]*float(gain[i]);
    float den=1.f+abs(float(raw[i])+45.f);
    float dvoltage=(fired ? 0.f : dv[i]) + (free ? ds/(den*den) : 0.f);
    if (silent) dvoltage=0.f;
    dg[i]=(fired ? 0.f : dg[i])*(free ? float(p[1]) : 1.f)
          + (free ? dvoltage*float(p[2]) : 0.f);
    dv[i]=dvoltage*(free ? float(p[0]) : 1.f);
    dr[i]*=float(p[4]);
}
'''
    return torch.mps.compile_shader(source.replace('ds/(den*den)',
        f'(ds*{float(surrogate_scale):.9e}f)/(den*den)'))


class NeuralWindow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factors, engine, drive, silence, ticks):
        b, lib = engine.brain, engine.lib
        n = len(b.ids)
        record = getattr(ctx, 'record_history', True)
        from .gesture_history import window_buffers
        raw, flags, emitted = window_buffers(engine, (ticks if record else 1, n), getattr(ctx, 'reuse_buffers', False))
        start = b.tick_index
        for t in range(ticks):
            h = t if record else 0
            slot = b.tick_index % len(b.delay_queue)
            future = (b.tick_index+b.delay_steps) % len(b.delay_queue)
            lib.integrate(b.voltage, b.current, b.spikes, b.last_spike_tick,
                b.refractory_ticks, drive, silence, b.output_gain, b.delay_queue[future],
                raw[h], flags[h], engine.coefficients, b.tick_index, threads=n)
            # Existing FP16 gather: identical edge order and half reductions.
            incoming = b.metal.deliver(b.delay_queue[slot])
            lib.finish(b.voltage, b.current, b.spikes, flags[h], incoming,
                b.delay_queue[slot], emitted[h], b.spike_counts, b.rates,
                engine.coefficients, threads=n)
            b.tick_index += 1
        b.time_ms = b.time_origin_ms+b.tick_index*b.config.dt_ms
        ctx.engine, ctx.start, ctx.ticks = engine, start, ticks
        ctx.save_for_backward(raw, flags, emitted, b.output_gain.clone())
        return b.rates.float()

    @staticmethod
    def backward(ctx, gradient):
        zero = torch.zeros_like(gradient)
        queue = torch.zeros_like(ctx.engine.brain.delay_queue, dtype=torch.float32)
        df, *_ = reverse_neural_window(ctx, gradient, zero, zero, queue)
        return df, None, None, None, None


def reverse_neural_window(ctx, gradient, voltage_grad, current_grad, queue_grad):
    e, b = ctx.engine, ctx.engine.brain
    raw, flags, emitted, gain = ctx.saved_tensors
    n = len(b.ids)
    dv, dg = voltage_grad.clone(), current_grad.clone()
    dr = gradient.contiguous().clone()
    dq = queue_grad.clone()
    di = torch.empty_like(gradient)
    df = torch.zeros((e.adapter.group_count, e.adapter.group_count), device='mps')
    partial = torch.empty(e.pair_chunks, device='mps', dtype=torch.float32)
    for t in reversed(range(ctx.ticks)):
        tick = ctx.start+t
        slot, future = tick % len(dq), (tick+b.delay_steps) % len(dq)
        e.lib.incoming_grad(dg, flags[t], di, threads=n)
        # Consumed queue entries replace, rather than accumulate with, their
        # post-consumption adjoint. Zero-delay emission is handled in order.
        e.lib.transpose_grad(e.out_ptr, e.out_post, e.out_weight, di, dq[slot],
                             threads=n*32, group_size=256)
        e.lib.active_factor_grad(e.pair_ptr, e.pair_source, e.pair_post, e.pair_base,
            emitted[t], di, partial, threads=e.pair_chunks*32, group_size=256)
        e.lib.reduce_factors(e.group_ptr, partial, df, threads=df.numel()*32, group_size=256)
        e.lib.reverse_state(dv, dg, dr, dq[future], gain, raw[t], flags[t],
                            e.coefficients, threads=n)
    return df, dv, dg, dr, dq


class MetalGradientBrain:
    def __init__(self, brain, adapter, bridge, surrogate_scale=1.):
        if brain.device.type != 'mps' or brain.dtype != torch.float16:
            raise ValueError('Metal training requires MPS float16 neural execution')
        self.brain, self.adapter, self.bridge = brain, adapter, bridge
        self.parameters = torch.nn.Parameter(torch.tensor(adapter.initial, device='mps', dtype=torch.float32))
        self.surrogate_scale = surrogate_scale
        self.lib = training_kernels(surrogate_scale)
        self.out_ptr = brain.out_ptr.to('mps', dtype=torch.int32)
        self.out_post = brain.out_post.to('mps', dtype=torch.int32)
        self.out_weight = brain.out_weight.to('mps', dtype=torch.float16)
        # One CSR row per adapter group pair: deterministic FP32 reductions,
        # without an edge-sized gradient tensor on every neural tick.
        order = np.argsort(adapter.out_pairs, kind='stable')
        counts = np.bincount(adapter.out_pairs, minlength=adapter.group_count**2)
        pre = np.repeat(np.arange(len(brain.ids), dtype=np.int32), np.diff(brain.out_ptr.numpy()))
        # Split large groups into bounded chunks to keep the GPU lanes busy.
        # Every edge is still visited; a second fixed-order reduction joins them.
        boundaries = np.r_[0, counts.cumsum()]
        sorted_pre = pre[order]
        # Each segment belongs to one presynaptic neuron and one group pair.
        # Zero emitted spikes then have exactly zero weight derivative, so the
        # kernel can skip that segment without changing connectivity or BPTT.
        source_boundaries = np.flatnonzero(np.diff(sorted_pre)) + 1
        bounded = np.concatenate([np.arange(start, end, 1024)
                                  for start, end in zip(boundaries[:-1], boundaries[1:])])
        chunk_ptr = np.unique(np.r_[bounded, source_boundaries, boundaries])
        self.pair_chunks = len(chunk_ptr)-1
        self.pair_ptr = torch.tensor(chunk_ptr, device='mps', dtype=torch.int32)
        self.pair_source = torch.tensor(sorted_pre[chunk_ptr[:-1]], device='mps', dtype=torch.int32)
        self.group_ptr = torch.tensor(np.searchsorted(chunk_ptr, boundaries), device='mps', dtype=torch.int32)
        self.pair_pre = torch.tensor(pre[order], device='mps', dtype=torch.int32)
        self.pair_post = torch.tensor(brain.out_post.numpy()[order], device='mps', dtype=torch.int32)
        self.pair_base = torch.tensor(adapter.base_out.numpy()[order], device='mps', dtype=torch.float32)
        c = brain.config
        a, b = math.exp(-c.dt_ms/c.membrane_ms), math.exp(-c.dt_ms/c.synapse_ms)
        coupling = (c.dt_ms/c.membrane_ms*a if c.synapse_ms == c.membrane_ms else
                    c.synapse_ms/(c.membrane_ms-c.synapse_ms)*(a-b))
        decay = math.exp(-c.dt_ms/50.)
        self.coefficients = torch.tensor([a,b,coupling,1-a,decay,(1-decay)*1000/c.dt_ms], device='mps', dtype=torch.float16)
        self.channels = [ids.to('mps') for ids in bridge.channels]
        self.channel_weights = [weights.to('mps', dtype=torch.float32) for weights in bridge.channel_weights]
        self.reset()

    def reset(self):
        self.history_sequence = []
        self.history_state = None
        self.rates = self.brain.rates.float()

    def detach_state(self):
        self.history_sequence = []
        self.history_state = None
        self.rates = self.rates.detach()

    def sync_weights(self):
        self.adapter.apply(self.parameters.detach().cpu().numpy())
        self.out_weight.copy_(self.brain.out_weight.to('mps', dtype=torch.float16))

    def advance(self, drive, duration_ms=20., silence=None):
        ticks = round(duration_ms/self.brain.config.dt_ms)
        if ticks < 1 or not math.isclose(ticks*self.brain.config.dt_ms, duration_ms):
            raise ValueError('Duration must cover complete neural ticks')
        u, v = self.parameters.unbind()
        factors = 1+.75*torch.tanh(u@v.T/math.sqrt(self.adapter.rank))
        drive = drive.to('mps', dtype=torch.float16)
        silence = (torch.zeros(len(self.brain.ids), device='mps', dtype=torch.bool)
                   if silence is None else silence.to('mps'))
        if getattr(self, 'full_history', False) and torch.is_grad_enabled():
            from .gesture_history import advance_full_history
            self.rates = advance_full_history(self, factors, drive, silence, ticks)
        elif not torch.is_grad_enabled():
            from types import SimpleNamespace
            ctx = SimpleNamespace(save_for_backward=lambda *values: None, record_history=False, reuse_buffers=True)
            self.rates = NeuralWindow.forward(ctx, factors, self, drive, silence, ticks)
        else:
            self.rates = NeuralWindow.apply(factors, self, drive, silence, ticks)
        self.brain.total_spikes = int(self.brain.spike_counts.sum())
        return self.rates

    def muscles(self):
        from .motor_mapping import LEGACY_PROFILE
        action = []
        for ids, weights in zip(self.channels, self.channel_weights):
            rate = ((self.rates[ids] if self.bridge.mapping_profile == LEGACY_PROFILE else
                     self.rates[ids]*weights).mean() if len(ids) else self.rates.sum()*0)
            action.append((rate/100*float(np.exp(self.bridge.parameters[7]))).clamp(0,1))
        return torch.stack(action)
