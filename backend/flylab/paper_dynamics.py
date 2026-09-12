"""Shiu et al. 2024 LIF dynamics, in mV and ms, with Brian2 scheduling.

Equations/parameters: https://doi.org/10.1038/s41586-024-07763-9
Reference code: philshiu/Drosophila_brain_model, commit 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960.
Both state variables freeze in refractory periods; synaptic writes into
refractory neurons are discarded, as in Brian2's ``unless refractory``.
"""
from __future__ import annotations

import math
import time
import torch

PAPER_DYNAMICS_VERSION = 'shiu-2024-linear-delay-v1'


class PaperDynamics:
    """Full-population state; analytic linear integration each 0.1 ms.

    Event delivery only visits edges of neurons that actually fired. All neurons
    still integrate on every tick; no activity threshold, pruning or top-k.
    """

    def prepare_paper(self):
        # Rows of this representation are presynaptic. Original CSR and integer
        # anatomical counts remain unchanged. No dense N-by-N matrix is made.
        outgoing = self.weights.transpose(0, 1).to_sparse_csr()
        self.out_ptr = outgoing.crow_indices()
        self.out_post = outgoing.col_indices()
        self.out_weight = outgoing.values()
        self.delay_steps = round(self.config.delay_ms / self.config.dt_ms)
        self.dtype = torch.float64
        self.device = torch.device('cpu')
        self.metal = None
        self.execution_history = []

    def set_device(self, device):
        """Migrate complete state atomically; MPS explicitly uses float32.

        Converting back to float64 cannot recover precision previously lost.
        RNG stays on CPU so identical seeds generate identical input events.
        """
        from .metal_dynamics import MetalDynamics, NEURAL_TENSORS
        if device not in {'cpu', 'mps'}:
            raise ValueError('Supported execution devices: cpu, mps')
        if self.config.profile != 'shiu-2024':
            raise ValueError('Device selection requires paper physiology')
        if self.device.type == device:
            return
        dtype = torch.float32 if device == 'mps' else torch.float64
        metal = MetalDynamics(self.weights) if device == 'mps' else None
        state = {name: getattr(self, name).to(device=device,
                 dtype=dtype if getattr(self, name).is_floating_point() else getattr(self, name).dtype)
                 for name in NEURAL_TENSORS}
        if device == 'mps':
            torch.mps.synchronize()
        self.execution_history.append({'from': self.device.type, 'to': device, 'tick': self.tick_index})
        self.device, self.dtype, self.metal = torch.device(device), dtype, metal
        for name, value in state.items():
            setattr(self, name, value)

    def execution_summary(self):
        from .metal_dynamics import METAL_VERSION
        gpu = self.voltage.device.type == 'mps'
        return {'device': self.voltage.device.type, 'precision': str(self.voltage.dtype),
                'kernel': METAL_VERSION if gpu else 'cpu-event-f64-v1',
                'experimental': gpu, 'history': self.execution_history,
                'note': 'MPS float32 may change spike timing; not equivalent to the float64 reference. All neurons, edges and ticks retained.' if gpu else 'CPU float64 reference. Prior float32 rounding, if any, is not reversible.'}

    def reset_paper(self, seed=0):
        n = len(self.ids)
        self.voltage = torch.full((n,), -52., dtype=self.dtype)
        self.current = torch.zeros(n, dtype=self.dtype)
        self.spikes = torch.zeros(n, dtype=self.dtype)
        self.rates = torch.zeros(n, dtype=self.dtype)
        self.output_gain = torch.ones(n, dtype=self.dtype)
        self.last_spike_tick = torch.full((n,), -10**12, dtype=torch.int64)
        self.refractory_ticks = torch.full((n,), round(self.config.refractory_ms / self.config.dt_ms), dtype=torch.int64)
        self.delay_queue = torch.zeros((self.delay_steps + 1, n), dtype=self.dtype)
        self.spike_counts = torch.zeros(n, dtype=torch.int64)
        self.generator = torch.Generator().manual_seed(seed)
        self.tick_index = 0
        self.time_ms = 0.
        self.total_spikes = 0
        self.last_wall_seconds = 0.
        self.execution_history = []
        if self.device.type != 'cpu':
            from .metal_dynamics import NEURAL_TENSORS
            for name in NEURAL_TENSORS:
                setattr(self, name, getattr(self, name).to(self.device))

    def deliver(self, emitted):
        if self.metal is not None:
            return self.metal.deliver(emitted)
        active = emitted.nonzero().flatten()
        result = torch.zeros_like(self.current)
        if not len(active):
            return result
        starts = self.out_ptr[active]
        lengths = self.out_ptr[active + 1] - starts
        total = int(lengths.sum())
        if total:
            offsets = torch.repeat_interleave(starts - (lengths.cumsum(0) - lengths), lengths)
            edges = offsets + torch.arange(total)
            amplitudes = torch.repeat_interleave(emitted[active], lengths)
            result.index_add_(0, self.out_post[edges], self.out_weight[edges] * amplitudes)
        return result

    @torch.no_grad()
    def advance_paper(self, drive, duration_ms=20., silence=None, *, voltage_events=None, poisson_hz=None):
        """Drive is constant injected current expressed as equivalent mV.

        Optional voltage_events[t, n] are deterministic input voltage jumps in
        the synapses phase. Poisson stimulation uses the paper's 68.75 mV jump;
        caller explicitly sets those receptor neurons' refractory_ticks to zero.
        ``silence`` is a soma clamp for workbench interventions; paper experiments
        instead set output_gain=0 to reproduce outgoing-synapse silencing.
        """
        c = self.config
        steps = round(duration_ms / c.dt_ms)
        if steps < 1 or not math.isclose(steps * c.dt_ms, duration_ms):
            raise ValueError('Duration must be a positive multiple of integration time')
        drive = torch.as_tensor(drive, dtype=self.dtype, device=self.device)
        if drive.shape != self.voltage.shape or not torch.isfinite(drive).all():
            raise ValueError('Drive must be finite and cover every neuron')
        if voltage_events is not None and (voltage_events.shape != (steps, len(self.ids)) or not torch.isfinite(voltage_events).all()):
            raise ValueError('Expected finite voltage events for every step and neuron')
        inputs = None
        if poisson_hz is not None:
            if poisson_hz.shape != drive.shape or not torch.isfinite(poisson_hz).all() or (poisson_hz < 0).any() or (poisson_hz * c.dt_ms > 1000).any():
                raise ValueError('Invalid Poisson input rates')
            inputs = poisson_hz.nonzero().flatten()
        if self.metal is not None:
            return self._advance_metal(drive, steps, silence, voltage_events, poisson_hz, inputs)
        started = time.perf_counter()
        a, b = math.exp(-c.dt_ms / c.membrane_ms), math.exp(-c.dt_ms / c.synapse_ms)
        coupling = c.dt_ms / c.membrane_ms * a if c.synapse_ms == c.membrane_ms else c.synapse_ms / (c.membrane_ms - c.synapse_ms) * (a - b)
        rate_decay = math.exp(-c.dt_ms / 50.)
        for step in range(steps):
            free = self.tick_index - self.last_spike_tick >= self.refractory_ticks
            v = -52. + (self.voltage + 52.) * a + self.current * coupling + drive * (1. - a)
            self.voltage.copy_(torch.where(free, v, self.voltage))
            self.current.copy_(torch.where(free, self.current * b, self.current))
            if silence is not None:
                self.voltage[silence] = -52.
            fired = free & (self.voltage > -45.)
            self.spikes.copy_(fired)
            self.last_spike_tick[fired] = self.tick_index
            free &= ~fired
            # Thresholds precede synaptic transmission, which precedes resets.
            slot = self.tick_index % len(self.delay_queue)
            self.delay_queue[(self.tick_index + self.delay_steps) % len(self.delay_queue)] += self.spikes * self.output_gain
            incoming = self.deliver(self.delay_queue[slot])
            self.delay_queue[slot].zero_()
            self.current.add_(incoming * free)
            if voltage_events is not None:
                self.voltage.add_(voltage_events[step] * free)
            if inputs is not None and len(inputs):
                events = torch.rand(len(inputs), generator=self.generator, dtype=self.dtype) < poisson_hz[inputs] * c.dt_ms / 1000.
                self.voltage[inputs] += events * free[inputs] * (c.gain * 250.)
            self.voltage[fired] = -52.
            self.current[fired] = 0.
            self.spike_counts.add_(fired)
            self.rates.mul_(rate_decay).add_(self.spikes, alpha=(1. - rate_decay) * 1000. / c.dt_ms)
            self.tick_index += 1
        self.total_spikes = int(self.spike_counts.sum())
        self.time_ms = self.tick_index * c.dt_ms
        self.last_wall_seconds = time.perf_counter() - started
        return self.rates

    def _advance_metal(self, drive, steps, silence, voltage_events, poisson_hz, inputs):
        """Keep all neural state on GPU; transfer external stimuli per batch."""
        c = self.config
        started = time.perf_counter()
        a, b = math.exp(-c.dt_ms / c.membrane_ms), math.exp(-c.dt_ms / c.synapse_ms)
        coupling = c.dt_ms / c.membrane_ms * a if c.synapse_ms == c.membrane_ms else c.synapse_ms / (c.membrane_ms - c.synapse_ms) * (a - b)
        decay = math.exp(-c.dt_ms / 50.)
        parameters = torch.tensor([a, b, coupling, 1-a, decay, (1-decay)*1000/c.dt_ms], device=self.device)
        silence = (torch.zeros(len(self.ids), dtype=torch.bool) if silence is None else silence).to(device=self.device, dtype=torch.bool)
        if silence.shape != drive.shape:
            raise ValueError('Silence mask must cover every neuron')
        if voltage_events is not None:
            voltage_events = voltage_events.to(device=self.device, dtype=self.dtype)
        # Draw float64 CPU uniforms for agreement with the reference RNG and for
        # checkpoint portability. Only external Poisson stimuli are CPU-generated.
        jumps = None
        if inputs is not None and len(inputs):
            inputs = inputs.cpu()
            rates = poisson_hz.cpu()[inputs]
            jumps = torch.stack([torch.rand(len(inputs), generator=self.generator, dtype=torch.float64)
                                 < rates*c.dt_ms/1000 for _ in range(steps)]).to(self.device)
            inputs = inputs.to(self.device)
        for step in range(steps):
            events = self.metal.zero if voltage_events is None else voltage_events[step]
            if jumps is not None:
                events = events.clone()
                events[inputs] += jumps[step] * (c.gain * 250.)
            self.metal.tick(self, drive, silence, events, parameters)
            self.tick_index += 1
        torch.mps.synchronize()
        self.total_spikes = int(self.spike_counts.sum())
        self.time_ms = self.tick_index*c.dt_ms
        self.last_wall_seconds = time.perf_counter() - started
        return self.rates
