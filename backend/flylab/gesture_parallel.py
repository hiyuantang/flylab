"""Shared-weight GPU batching: independent neural states, one adapter gradient.

Every dispatch spans sample and neuron dimensions. Time remains sequential.
The single-sample kernels provide identical finish and state-adjoint arithmetic.
"""
from functools import lru_cache
import math
import torch
from .gesture_metal import training_kernels


@lru_cache(maxsize=8)
def batch_kernels(batch_size=1):
    source = r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void integrate(device half* v, device half* g, device half* spikes,
    device long* last, const device long* refractory, const device half* drive,
    const device bool* silence, const device half* gain, device half* future,
    device half* raw, device int* flags, device atomic_uint* active,
    const device half* p, constant long& tick, constant long& n,
    uint i [[thread_position_in_grid]]) {
    bool free = tick-last[i] >= refractory[i%n];
    if (free) { v[i] = -52.h + (v[i]+52.h)*p[0] + g[i]*p[2] + drive[i]*p[3]; g[i] *= p[1]; }
    if (silence[i]) v[i] = -52.h;
    bool fired = free && v[i] > -45.h;
    raw[i] = v[i]; flags[i] = int(free) | (int(fired)<<1) | (int(silence[i])<<2);
    spikes[i] = half(fired); if (fired) last[i] = tick;
    future[i] += half(fired)*gain[i];
    atomic_store_explicit(&active[i],0u,memory_order_relaxed);
}
kernel void mark(const device int* ptr, const device int* post,
    const device half* emitted, device atomic_uint* active, constant long& n,
    uint tid [[thread_position_in_grid]]) {
    uint i=tid/4, lane=tid%4, row=i%n, base=i-row;
    if (emitted[i]==0.h) return;
    for (int e=ptr[row]+int(lane);e<ptr[row+1];e+=4)
        atomic_store_explicit(&active[base+post[e]],1u,memory_order_relaxed);
}
kernel void gather(const device int* ptr, const device int* pre,
    const device half* weight, const device half* emitted, device half* incoming,
    const device atomic_uint* active, constant long& n,
    uint tid [[thread_position_in_grid]]) {
    uint i=tid/32,lane=tid%32,row=i%n,base=i-row;
    if (atomic_load_explicit(&active[i],memory_order_relaxed)==0u) {
        if (lane==0) incoming[i]=0.h; return;
    }
    half sum=0.h,error=0.h;
    for (int e=ptr[row]+int(lane);e<ptr[row+1];e+=32) {
        half value=emitted[base+pre[e]]*weight[e];
        half y=value-error,next=sum+y; error=(next-sum)-y;sum=next;
    }
    half reduced=simd_sum(sum);if(lane==0)incoming[i]=reduced;
}
kernel void transpose_grad(const device int* ptr, const device int* post,
    const device half* weight, const device float* di, device float* dq,
    constant long& n, uint tid [[thread_position_in_grid]]) {
    uint i=tid/32,lane=tid%32,row=i%n,base=i-row;float sum=0.f;
    for(int e=ptr[row]+int(lane);e<ptr[row+1];e+=32)sum+=float(weight[e])*di[base+post[e]];
    float reduced=simd_sum(sum);if(lane==0)dq[i]=reduced;
}
kernel void factor_grad(const device int* ptr, const device int* pre,
    const device int* post, const device float* base, const device half* emitted,
    const device float* di, device float* df, constant long& chunks, constant long& n,
    uint tid [[thread_position_in_grid]]) {
    uint i=tid/32,lane=tid%32,row=i%chunks,offset=(i/chunks)*n;float sum=0.f;
    for(int e=ptr[row]+int(lane);e<ptr[row+1];e+=32)
        sum+=base[e]*float(emitted[offset+pre[e]])*di[offset+post[e]];
    float reduced=simd_sum(sum);if(lane==0)df[i]=reduced;
}
kernel void reduce_factors(const device int* ptr, const device float* partial,
    device float* df, constant long& groups, constant long& chunks,
    uint tid [[thread_position_in_grid]]) {
    uint i=tid/32,lane=tid%32,row=i%groups,offset=(i/groups)*chunks;float sum=0.f;
    for(int e=ptr[row]+int(lane);e<ptr[row+1];e+=32)sum+=partial[offset+e];
    float reduced=simd_sum(sum);if(lane==0)df[i]+=reduced;
}
'''
    # One edge/index read serves every lane. Each sample retains its own sum;
    # no cross-sample neural state or gradient enters these reductions.
    sums = '\n'.join(f'float sum{b}=0.f;' for b in range(batch_size))
    updates = '\n'.join(f'sum{b}+=w*di[{b}*n+target];' for b in range(batch_size))
    writes = '\n'.join(f'float r{b}=simd_sum(sum{b});if(lane==0)dq[{b}*n+i]=r{b};' for b in range(batch_size))
    source += f"""
kernel void shared_transpose_grad(const device int* ptr, const device int* post,
    const device half* weight, const device float* di, device float* dq,
    constant long& n, uint tid [[thread_position_in_grid]]) {{
    uint i=tid/32,lane=tid%32; {sums}
    for(int e=ptr[i]+int(lane);e<ptr[i+1];e+=32){{
        int target=post[e];float w=float(weight[e]);{updates}
    }}
    {writes}
}}
"""
    reads = '\n'.join(f'float q{b}=float(emitted[{b}*n+pre]);' for b in range(batch_size))
    active = '||'.join(f'q{b}!=0.f' for b in range(batch_size))
    updates = '\n'.join(f'if(q{b}!=0.f)sum{b}+=w*q{b}*di[{b}*n+target];' for b in range(batch_size))
    writes = '\n'.join(f'float r{b}=simd_sum(sum{b});if(lane==0)df[{b}*chunks+i]=r{b};' for b in range(batch_size))
    source += f"""
kernel void active_factor_grad(const device int* ptr, const device int* source,
    const device int* post, const device float* base, const device half* emitted,
    const device float* di, device float* df, constant long& chunks, constant long& n,
    uint tid [[thread_position_in_grid]]) {{
    uint i=tid/32,lane=tid%32;int pre=source[i];{reads} {sums}
    if({active}){{
        for(int e=ptr[i]+int(lane);e<ptr[i+1];e+=32){{
            int target=post[e];float w=base[e];{updates}
        }}
    }}
    {writes}
}}
"""
    return torch.mps.compile_shader(source)


class BatchWindow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factors, engine, drive, silence, ticks):
        e, s = engine, engine.shared
        record = getattr(ctx, 'record_history', True)
        from .gesture_history import window_buffers
        raw,flags,emitted = window_buffers(e,(ticks if record else 1,e.batch_size,e.n),getattr(ctx,'reuse_buffers',False))
        start = e.tick_index
        total = e.batch_size*e.n
        for t in range(ticks):
            h = t if record else 0
            slot = e.tick_index%len(e.delay_queue)
            future = (e.tick_index+e.delay_steps)%len(e.delay_queue)
            e.lib.integrate(e.voltage,e.current,e.spikes,e.last_spike_tick,
                s.brain.refractory_ticks,drive,silence,e.output_gain,e.delay_queue[future],
                raw[h],flags[h],e.active,s.coefficients,e.tick_index,e.n,threads=total)
            e.lib.mark(s.out_ptr,s.out_post,e.delay_queue[slot],e.active,e.n,threads=total*4)
            e.lib.gather(s.brain.metal.ptr,s.brain.metal.pre,s.brain.metal.weight,
                e.delay_queue[slot],e.incoming,e.active,e.n,threads=total*32,group_size=256)
            e.common.finish(e.voltage,e.current,e.spikes,flags[h],e.incoming,
                e.delay_queue[slot],emitted[h],e.spike_counts,e.rates,s.coefficients,threads=total)
            e.tick_index+=1
        ctx.engine,ctx.start,ctx.ticks=e,start,ticks
        ctx.save_for_backward(raw,flags,emitted,e.output_gain.clone())
        return e.rates.float()

    @staticmethod
    def backward(ctx, gradient):
        zero = torch.zeros_like(gradient)
        queue = torch.zeros_like(ctx.engine.delay_queue, dtype=torch.float32)
        df, *_ = reverse_batch_window(ctx, gradient, zero, zero, queue)
        return df, None, None, None, None


def reverse_batch_window(ctx, gradient, voltage_grad, current_grad, queue_grad):
    e,s=ctx.engine,ctx.engine.shared
    raw,flags,emitted,gain=ctx.saved_tensors
    total=e.batch_size*e.n
    dv,dg=voltage_grad.clone(),current_grad.clone()
    dr=gradient.contiguous().clone()
    dq=queue_grad.clone()
    di=torch.empty_like(gradient)
    df=torch.zeros((e.batch_size,s.adapter.group_count**2),device='mps')
    partial=torch.empty((e.batch_size,s.pair_chunks),device='mps',dtype=torch.float32)
    for t in reversed(range(ctx.ticks)):
        tick=ctx.start+t
        slot,future=tick%len(dq),(tick+e.delay_steps)%len(dq)
        e.common.incoming_grad(dg,flags[t],di,threads=total)
        e.lib.shared_transpose_grad(s.out_ptr,s.out_post,s.out_weight,di,dq[slot],e.n,
                             threads=e.n*32,group_size=256)
        e.lib.active_factor_grad(s.pair_ptr,s.pair_source,s.pair_post,s.pair_base,emitted[t],
            di,partial,s.pair_chunks,e.n,threads=s.pair_chunks*32,group_size=256)
        e.lib.reduce_factors(s.group_ptr,partial,df,s.adapter.group_count**2,
            s.pair_chunks,threads=df.numel()*32,group_size=256)
        e.common.reverse_state(dv,dg,dr,dq[future],gain,raw[t],flags[t],s.coefficients,threads=total)
    return df.sum(dim=0).reshape(s.adapter.group_count,s.adapter.group_count), dv, dg, dr, dq


class ParallelMetalBrain:
    STATE_NAMES=('voltage','current','spikes','rates','output_gain','last_spike_tick','spike_counts')

    def __init__(self,shared,batch_size):
        if not isinstance(batch_size,int) or not 1<=batch_size<=64:
            raise ValueError('Parallel batch size must be from 1 to 64')
        self.shared,self.batch_size=shared,batch_size
        self.n=len(shared.brain.ids)
        # Bound the known neural working set before allocating all histories.
        per_sample=self.n*(200*8+160)
        available=.9*torch.mps.recommended_max_memory()-torch.mps.current_allocated_memory()
        if per_sample*batch_size>available:
            raise ValueError(f'Parallel batch needs about {per_sample*batch_size/1e9:.1f} GB additional GPU memory; reduce batch size')
        self.lib,self.common=batch_kernels(batch_size),shared.lib
        self.active=torch.zeros((batch_size,self.n),device='mps',dtype=torch.int32)
        self.incoming=torch.empty((batch_size,self.n),device='mps',dtype=torch.float16)
        self.reset()

    def reset(self):
        b=self.shared.brain
        for name in self.STATE_NAMES:
            value=getattr(b,name)
            setattr(self,name,value.unsqueeze(0).expand(self.batch_size,-1).clone())
        self.delay_queue=b.delay_queue.unsqueeze(1).expand(-1,self.batch_size,-1).clone()
        self.tick_index,self.delay_steps=b.tick_index,b.delay_steps
        self.history_sequence=[]
        self.history_state=None
        self.output_rates=self.rates.float()

    def advance(self,drive,silence=None,duration_ms=20.):
        ticks=round(duration_ms/self.shared.brain.config.dt_ms)
        if ticks<1 or not math.isclose(ticks*self.shared.brain.config.dt_ms,duration_ms):
            raise ValueError('Duration must cover complete neural ticks')
        if drive.shape!=(self.batch_size,self.n):
            raise ValueError('Each batch sample needs its own neural input')
        drive=drive.to('mps',dtype=torch.float16).contiguous()
        silence=(torch.zeros_like(drive,dtype=torch.bool) if silence is None else silence.to('mps').contiguous())
        if silence.shape!=drive.shape:raise ValueError('Batch silence shape mismatch')
        u,v=self.shared.parameters.unbind()
        factors=1+.75*torch.tanh(u@v.T/math.sqrt(self.shared.adapter.rank))
        if getattr(self.shared, 'full_history', False) and torch.is_grad_enabled():
            from .gesture_history import advance_full_history
            self.output_rates=advance_full_history(self,factors,drive,silence,ticks)
        elif not torch.is_grad_enabled():
            from types import SimpleNamespace
            ctx = SimpleNamespace(save_for_backward=lambda *values: None, record_history=False, reuse_buffers=True)
            self.output_rates = BatchWindow.forward(ctx, factors, self, drive, silence, ticks)
        else:
            self.output_rates=BatchWindow.apply(factors,self,drive,silence,ticks)
        return self.output_rates

    def muscles(self):
        from .motor_mapping import LEGACY_PROFILE
        import numpy as np
        s=self.shared
        action=[]
        for ids,weights in zip(s.channels,s.channel_weights):
            rate=((self.output_rates[:,ids] if s.bridge.mapping_profile==LEGACY_PROFILE else
                self.output_rates[:,ids]*weights).mean(dim=1) if len(ids) else self.output_rates.sum(dim=1)*0)
            action.append((rate/100*float(np.exp(s.bridge.parameters[7]))).clamp(0,1))
        return torch.stack(action,dim=1)

    def detach_state(self):
        self.history_sequence=[]
        self.history_state=None
        self.output_rates=self.output_rates.detach()

    def export_sample(self,index):
        """Copy one lane into the existing preview brain; graph storage is shared."""
        if not 0<=index<self.batch_size:raise ValueError('Sample index is outside the batch')
        b=self.shared.brain
        for name in self.STATE_NAMES:getattr(b,name).copy_(getattr(self,name)[index])
        b.delay_queue.copy_(self.delay_queue[:,index,:])
        b.tick_index=self.tick_index
        b.time_ms=b.time_origin_ms+b.tick_index*b.config.dt_ms
        b.total_spikes=int(b.spike_counts.sum())
        self.shared.reset()
