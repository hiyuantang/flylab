"""Full-graph Metal execution. FP16 weight storage is an opt-in experiment.

Mixed mode keeps neural state and arithmetic in float32; all-FP16 mode also
rounds state and arithmetic. Optimized active-row FP16 is an explicit live option.
CPU float64 remains reference; entering GPU without a precision defaults to FP32.

The baseline uses three ordered dispatches per tick for threshold, synapse and
reset phases; active-row mode adds an integer marking pass. CSR gather retains
every edge and avoids unsupported MPS sparse
operators and floating-point atomic races. No neurons or ticks are skipped.
"""
from functools import lru_cache
import re
import torch

METAL_VERSION = 'metal-csr-f32-v1'
MIXED_METAL_VERSION = 'metal-csr-w16-state32-v1'
HALF_METAL_VERSION = 'metal-csr-f16-v1'
ACTIVE_HALF_METAL_VERSION = 'metal-active-rows-f16-v1'
FUSED_HALF_METAL_VERSION = 'metal-active-fused-f16-v1'
NEURAL_TENSORS = ('voltage', 'current', 'spikes', 'rates', 'output_gain',
                  'last_spike_tick', 'refractory_ticks', 'delay_queue', 'spike_counts')


@lru_cache(maxsize=12)
def kernels(weight_dtype=torch.float32, state_dtype=torch.float32, active_rows=False, fused=False, mark_lanes=1):
    if weight_dtype not in {torch.float32, torch.float16}:
        raise ValueError("Metal weights require float32 or float16")
    if state_dtype not in {torch.float32, torch.float16}:
        raise ValueError("Metal state requires float32 or float16")
    if state_dtype == torch.float16 and weight_dtype != torch.float16:
        raise ValueError("FP16 state experiment requires FP16 weights")
    source = r'''
    #include <metal_stdlib>
    using namespace metal;
    #pragma clang fp contract(off)
    kernel void integrate(device float* v, device float* g,
        device float* spikes, device long* last, const device long* refractory,
        const device float* drive, const device bool* silence,
        const device float* gain, device float* future, device int* free,
        constant long& tick, const device float* p,
        uint i [[thread_position_in_grid]]) {
        bool ready = tick - last[i] >= refractory[i];
        if (ready) {
            v[i] = -52.f + (v[i] + 52.f)*p[0] + g[i]*p[2] + drive[i]*p[3];
            g[i] *= p[1];
        }
        if (silence[i]) v[i] = -52.f;
        bool fired = ready && v[i] > -45.f;
        spikes[i] = float(fired);
        if (fired) last[i] = tick;
        free[i] = int(ready && !fired);
        future[i] += float(fired)*gain[i];
    }
    kernel void gather(const device int* ptr, const device int* pre,
        const device WEIGHT_TYPE* weight, const device float* emitted,
        device float* incoming, uint tid [[thread_position_in_grid]]) {
        uint i = tid / 32, lane = tid % 32;
        float sum = 0.f;
        // Fixed lane/reduction order; compensated partial sums limit error.
        float error = 0.f;
        for (int e = ptr[i] + int(lane); e < ptr[i+1]; e += 32) {
            float value = emitted[pre[e]] * float(weight[e]);
            float y = value - error;
            float next = sum + y;
            error = (next - sum) - y;
            sum = next;
        }
        float reduced = simd_sum(sum);
        if (lane == 0) incoming[i] = reduced;
    }
    kernel void finish(device float* v, device float* g,
        const device float* spikes, const device int* free,
        const device float* incoming, const device float* events,
        device float* consumed, device long* counts, device float* rates,
        const device float* p, uint i [[thread_position_in_grid]]) {
        g[i] += incoming[i]*float(free[i]);
        v[i] += events[i]*float(free[i]);
        if (spikes[i] != 0.f) { v[i] = -52.f; g[i] = 0.f; }
        consumed[i] = 0.f;
        counts[i] += long(spikes[i]);
        rates[i] = rates[i]*p[4] + spikes[i]*p[5];
    }
    '''
    if active_rows:
        # Integer flags only: no floating-point atomic accumulation or reordered sums.
        source = source.replace('const device float* p,\n        uint i',
                                'const device float* p, device atomic_uint* active,\n        uint i', 1)
        source = source.replace('future[i] += float(fired)*gain[i];',
                                'future[i] += float(fired)*gain[i];\n        atomic_store_explicit(&active[i], 0u, memory_order_relaxed);')
        source = source.replace('device float* incoming, uint tid',
                                'device float* incoming, device atomic_uint* active, uint tid')
        source = source.replace('uint i = tid / 32, lane = tid % 32;',
            'uint i = tid / 32, lane = tid % 32;\n'
            '        if (atomic_load_explicit(&active[i], memory_order_relaxed) == 0u) {\n'
            '            if (lane == 0) incoming[i] = 0.f;\n'
            '            return;\n'
            '        }')
        source += r'''
        kernel void mark_active(const device int* ptr, const device int* post,
            const device float* emitted, device atomic_uint* active,
            uint i [[thread_position_in_grid]]) {
            if (emitted[i] == 0.f) return;
            for (int e = ptr[i]; e < ptr[i+1]; ++e)
                atomic_store_explicit(&active[post[e]], 1u, memory_order_relaxed);
        }
        '''
    if fused:
        if not active_rows:
            raise ValueError('Fused delivery requires active rows')
        # Snapshot this tick's delayed signals before clearing the queue. This
        # avoids reading a buffer other SIMD groups are concurrently clearing.
        # For zero delay, future==consumed: emission precedes this snapshot.
        source = source.replace('const device float* p, device atomic_uint* active,\n        uint i',
            'const device float* p, device atomic_uint* active,\n'
            '        device float* emitted, device float* consumed,\n        uint i', 1)
        source = source.replace('atomic_store_explicit(&active[i], 0u, memory_order_relaxed);',
            'atomic_store_explicit(&active[i], 0u, memory_order_relaxed);\n'
            '        emitted[i] = consumed[i];\n        consumed[i] = 0.f;')
        source = source.replace('device float* incoming, device atomic_uint* active, uint tid',
            'device float* incoming, device atomic_uint* active,\n'
            '        device float* v, device float* g, const device float* spikes,\n'
            '        const device int* free, const device float* events, device long* counts,\n'
            '        device float* rates, const device float* p, uint tid')
        source = source.replace(
            'if (atomic_load_explicit(&active[i], memory_order_relaxed) == 0u) {\n'
            '            if (lane == 0) incoming[i] = 0.f;\n'
            '            return;\n'
            '        }',
            'bool has_input = atomic_load_explicit(&active[i], memory_order_relaxed) != 0u;')
        source = source.replace('e < ptr[i+1]; e += 32', 'has_input && e < ptr[i+1]; e += 32')
        source = source.replace('if (lane == 0) incoming[i] = reduced;', r'''
        if (lane == 0) {
            incoming[i] = reduced;
            g[i] += reduced*float(free[i]);
            v[i] += events[i]*float(free[i]);
            if (spikes[i] != 0.f) { v[i] = -52.f; g[i] = 0.f; }
            counts[i] += long(spikes[i]);
            rates[i] = rates[i]*p[4] + spikes[i]*p[5];
        }
        ''')
    if mark_lanes not in {1, 4, 8, 32}:
        raise ValueError('Marking lanes must be 1, 4, 8 or 32')
    if active_rows and mark_lanes != 1:
        source = source.replace('uint i [[thread_position_in_grid]]) {\n            if (emitted[i] == 0.f)',
            f'uint tid [[thread_position_in_grid]]) {{\n            uint i = tid / {mark_lanes}, lane = tid % {mark_lanes};\n            if (emitted[i] == 0.f)')
        source = source.replace('for (int e = ptr[i]; e < ptr[i+1]; ++e)',
                                f'for (int e = ptr[i] + int(lane); e < ptr[i+1]; e += {mark_lanes})')
    source = source.replace('WEIGHT_TYPE', 'half' if weight_dtype == torch.float16 else 'float')
    if state_dtype == torch.float16:
        # Half pointers, locals, casts AND literals avoid implicit FP32 promotion.
        # Graph indices and integer timing/count state retain their exact types.
        source = re.sub(r'\bfloat\b', 'half', source)
        source = re.sub(r'(\d+\.)f\b', r'\1h', source)
    return torch.mps.compile_shader(source)


class MetalDynamics:
    def __init__(self, weights, *, weight_dtype=torch.float32, state_dtype=torch.float32):
        if not torch.backends.mps.is_available():
            raise ValueError('Apple MPS GPU is unavailable in this process')
        if not hasattr(torch.mps, 'compile_shader'):
            raise ValueError('This GPU backend requires torch.mps.compile_shader')
        if max(weights.shape) >= 2**31 or weights._nnz() >= 2**31:
            raise ValueError('Metal CSR index range exceeded; graph was not truncated')
        if weight_dtype not in {torch.float32, torch.float16}:
            raise ValueError('Metal weights require float32 or float16')
        if state_dtype not in {torch.float32, torch.float16}:
            raise ValueError('Metal state requires float32 or float16')
        if state_dtype == torch.float16 and weight_dtype != torch.float16:
            raise ValueError('FP16 state experiment requires FP16 weights')
        stored = weights.values().to(device='cpu', dtype=torch.float32)
        if weight_dtype == torch.float16:
            rounded = stored.to(torch.float16)
            if not torch.isfinite(rounded).all():
                raise ValueError('FP16 weight overflow or nonfinite value; experiment refused')
            if ((stored != 0) & (rounded == 0)).any():
                raise ValueError('FP16 weight underflow would erase a nonzero efficacy; experiment refused')
            stored = rounded
        self.weight_dtype = weight_dtype
        self.state_dtype = state_dtype
        self.version = (HALF_METAL_VERSION if state_dtype == torch.float16 else
                        MIXED_METAL_VERSION if weight_dtype == torch.float16 else METAL_VERSION)
        self.lib = kernels(weight_dtype, state_dtype)
        self.ptr = weights.crow_indices().to(device='mps', dtype=torch.int32)
        self.pre = weights.col_indices().to(device='mps', dtype=torch.int32)
        self.weight = stored.to(device='mps')
        self.n = weights.shape[0]
        self.incoming = torch.empty(self.n, device='mps', dtype=state_dtype)
        self.free = torch.empty(self.n, dtype=torch.int32, device='mps')
        self.zero = torch.zeros(self.n, device='mps', dtype=state_dtype)
        torch.mps.synchronize()

    def deliver(self, emitted):
        if emitted.dtype != self.state_dtype:
            raise ValueError('Signal precision does not match the Metal kernel')
        self.lib.gather(self.ptr, self.pre, self.weight, emitted,
                        self.incoming, threads=self.n*32, group_size=256)
        return self.incoming

    def tick(self, brain, drive, silence, events, parameters):
        if any(t.dtype != self.state_dtype for t in (brain.voltage, brain.current, drive, events, parameters)):
            raise ValueError('State or input precision does not match the Metal kernel')
        slot = brain.tick_index % len(brain.delay_queue)
        future = (brain.tick_index + brain.delay_steps) % len(brain.delay_queue)
        self.lib.integrate(brain.voltage, brain.current, brain.spikes,
                           brain.last_spike_tick, brain.refractory_ticks, drive,
                           silence, brain.output_gain, brain.delay_queue[future],
                           self.free, brain.tick_index, parameters, threads=self.n)
        self.deliver(brain.delay_queue[slot])
        self.lib.finish(brain.voltage, brain.current, brain.spikes, self.free,
                        self.incoming, events, brain.delay_queue[slot],
                        brain.spike_counts, brain.rates, parameters, threads=self.n)


class ActiveRowMetalDynamics(MetalDynamics):
    """FP16 candidate: gather only rows that receive a nonzero delayed signal.

    All neurons still integrate and finish each tick. A marked row uses the exact
    baseline CSR order, lane assignment and compensated sum, including zero terms.
    Flags are scratch state, cleared before each mark pass, not checkpoint state.
    """
    def __init__(self, weights, *, mark_lanes=4):
        super().__init__(weights, weight_dtype=torch.float16, state_dtype=torch.float16)
        self.lib = kernels(torch.float16, torch.float16, active_rows=True, mark_lanes=mark_lanes)
        outgoing = weights.transpose(0, 1).to_sparse_csr()
        self.out_ptr = outgoing.crow_indices().to(device='mps', dtype=torch.int32)
        self.out_post = outgoing.col_indices().to(device='mps', dtype=torch.int32)
        self.active = torch.zeros(self.n, device='mps', dtype=torch.int32)
        self.mark_lanes = mark_lanes
        self.version = ACTIVE_HALF_METAL_VERSION + (f'-mark{mark_lanes}' if mark_lanes != 1 else '')
        torch.mps.synchronize()

    def deliver(self, emitted, *, flags_cleared=False):
        if emitted.dtype != self.state_dtype:
            raise ValueError('Signal precision does not match the Metal kernel')
        if not flags_cleared:
            self.active.zero_()
        self.lib.mark_active(self.out_ptr, self.out_post, emitted, self.active, threads=self.n*self.mark_lanes)
        self.lib.gather(self.ptr, self.pre, self.weight, emitted, self.incoming,
                        self.active, threads=self.n*32, group_size=256)
        return self.incoming

    def tick(self, brain, drive, silence, events, parameters):
        if any(t.dtype != self.state_dtype for t in (brain.voltage, brain.current, drive, events, parameters)):
            raise ValueError('State or input precision does not match the Metal kernel')
        slot = brain.tick_index % len(brain.delay_queue)
        future = (brain.tick_index + brain.delay_steps) % len(brain.delay_queue)
        self.lib.integrate(brain.voltage, brain.current, brain.spikes,
                           brain.last_spike_tick, brain.refractory_ticks, drive,
                           silence, brain.output_gain, brain.delay_queue[future],
                           self.free, brain.tick_index, parameters, self.active, threads=self.n)
        self.deliver(brain.delay_queue[slot], flags_cleared=True)
        self.lib.finish(brain.voltage, brain.current, brain.spikes, self.free,
                        self.incoming, events, brain.delay_queue[slot],
                        brain.spike_counts, brain.rates, parameters, threads=self.n)


class FusedActiveRowMetalDynamics(ActiveRowMetalDynamics):
    """Three-pass active FP16: integrate/snapshot, mark, gather/finalize.

    The immutable emitted snapshot prevents inter-row queue-clear races. Neural
    states at tick boundaries and within-row arithmetic follow the FP16 baseline.
    The inherited deliver() remains available for standalone adjacency checks.
    """
    def __init__(self, weights, *, mark_lanes=1):
        super().__init__(weights, mark_lanes=mark_lanes)
        self.fused_lib = kernels(torch.float16, torch.float16, active_rows=True, fused=True, mark_lanes=mark_lanes)
        self.emitted = torch.empty(self.n, dtype=torch.float16, device='mps')
        self.mark_lanes = mark_lanes
        self.version = FUSED_HALF_METAL_VERSION + (f'-mark{mark_lanes}' if mark_lanes != 1 else '')
        torch.mps.synchronize()

    def tick(self, brain, drive, silence, events, parameters):
        if any(t.dtype != self.state_dtype for t in (brain.voltage, brain.current, drive, events, parameters)):
            raise ValueError('State or input precision does not match the Metal kernel')
        slot = brain.tick_index % len(brain.delay_queue)
        future = (brain.tick_index + brain.delay_steps) % len(brain.delay_queue)
        self.fused_lib.integrate(brain.voltage, brain.current, brain.spikes,
                           brain.last_spike_tick, brain.refractory_ticks, drive,
                           silence, brain.output_gain, brain.delay_queue[future],
                           self.free, brain.tick_index, parameters, self.active,
                           self.emitted, brain.delay_queue[slot], threads=self.n)
        self.fused_lib.mark_active(self.out_ptr, self.out_post, self.emitted, self.active, threads=self.n*self.mark_lanes)
        self.fused_lib.gather(self.ptr, self.pre, self.weight, self.emitted, self.incoming,
                             self.active, brain.voltage, brain.current, brain.spikes,
                             self.free, events, brain.spike_counts, brain.rates,
                             parameters, threads=self.n*32, group_size=256)
