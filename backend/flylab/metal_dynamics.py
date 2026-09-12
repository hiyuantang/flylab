"""Full-graph Metal execution. Float32 is explicit; CPU float64 remains reference.

Three ordered dispatches per tick implement the reference's threshold, synapse,
and reset phases. CSR gather retains every edge and avoids unsupported MPS sparse
operators and floating-point atomic races. No neurons or ticks are skipped.
"""
from functools import lru_cache
import torch

METAL_VERSION = 'metal-csr-f32-v1'
NEURAL_TENSORS = ('voltage', 'current', 'spikes', 'rates', 'output_gain',
                  'last_spike_tick', 'refractory_ticks', 'delay_queue', 'spike_counts')


@lru_cache(maxsize=1)
def kernels():
    return torch.mps.compile_shader(r'''
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
        const device float* weight, const device float* emitted,
        device float* incoming, uint tid [[thread_position_in_grid]]) {
        uint i = tid / 32, lane = tid % 32;
        float sum = 0.f;
        // Fixed lane/reduction order; compensated partial sums limit error.
        float error = 0.f;
        for (int e = ptr[i] + int(lane); e < ptr[i+1]; e += 32) {
            float value = emitted[pre[e]] * weight[e];
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
    ''')


class MetalDynamics:
    def __init__(self, weights):
        if not torch.backends.mps.is_available():
            raise ValueError('Apple MPS GPU is unavailable in this process')
        if not hasattr(torch.mps, 'compile_shader'):
            raise ValueError('This GPU backend requires torch.mps.compile_shader')
        if max(weights.shape) >= 2**31 or weights._nnz() >= 2**31:
            raise ValueError('Metal CSR index range exceeded; graph was not truncated')
        self.lib = kernels()
        self.ptr = weights.crow_indices().to(device='mps', dtype=torch.int32)
        self.pre = weights.col_indices().to(device='mps', dtype=torch.int32)
        self.weight = weights.values().to(device='mps', dtype=torch.float32)
        self.n = weights.shape[0]
        self.incoming = torch.empty(self.n, device='mps')
        self.free = torch.empty(self.n, dtype=torch.int32, device='mps')
        self.zero = torch.zeros(self.n, device='mps')
        torch.mps.synchronize()

    def deliver(self, emitted):
        self.lib.gather(self.ptr, self.pre, self.weight, emitted,
                        self.incoming, threads=self.n*32, group_size=256)
        return self.incoming

    def tick(self, brain, drive, silence, events, parameters):
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
