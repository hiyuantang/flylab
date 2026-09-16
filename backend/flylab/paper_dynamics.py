"""Shiu et al. 2024 LIF dynamics, in mV and ms, with Brian2 scheduling.

Equations/parameters: https://doi.org/10.1038/s41586-024-07763-9
Reference code: philshiu/Drosophila_brain_model, commit 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960.
Both state variables freeze in refractory periods; synaptic writes into
refractory neurons are discarded, as in Brian2's ``unless refractory``.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
import torch

PAPER_DYNAMICS_VERSION = 'shiu-2024-linear-delay-v1'


class PaperDynamics:
    """Full-population state; analytic integration on the configured neural clock.

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
        self.delay_steps = self.config.integration_ticks(self.config.delay_ms)
        self.dtype = torch.float64
        self.device = torch.device('cpu')
        self.metal = None
        self.execution_history = []

    def set_device(self, device, precision=None):
        """Migrate all persistent state, retaining clocks and CPU input RNG.

        Precision is explicit when changed on the same device. Device-only
        requests preserve its current precision; entering MPS defaults to FP32.
        Converting back cannot recover previously rounded values.
        """
        from .metal_dynamics import MetalDynamics, ActiveRowMetalDynamics, NEURAL_TENSORS
        if device not in {'cpu', 'mps'}:
            raise ValueError('Supported execution devices: cpu, mps')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('Apple MPS GPU is unavailable in this process')
        if self.config.profile != 'shiu-2024':
            raise ValueError('Device selection requires paper physiology')
        precision = precision or (str(self.dtype).removeprefix('torch.') if self.device.type == device
                                  else 'float32' if device == 'mps' else 'float64')
        if precision not in ({'float16', 'float32'} if device == 'mps' else {'float64'}):
            raise ValueError('MPS supports float16/float32; CPU requires float64')
        dtype = getattr(torch, precision)
        if self.device.type == device and self.dtype == dtype:
            return
        # Validate conversion on CPU before allocating a replacement GPU kernel.
        # Integer counters, queued event indices and neuron IDs remain integers.
        state = {}
        for name in NEURAL_TENSORS:
            previous = getattr(self, name)
            value = previous.cpu().to(dtype=dtype if previous.is_floating_point() else previous.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f'Precision conversion overflowed {name}; execution unchanged')
            state[name] = value.to(device=device)
        metal = (ActiveRowMetalDynamics(self.weights) if dtype == torch.float16 else MetalDynamics(self.weights)) if device == 'mps' else None
        if device == 'mps':
            torch.mps.synchronize()
        self.execution_history.append({'from': self.device.type, 'to': device, 'tick': self.tick_index,
                                       'from_precision': str(self.dtype), 'to_precision': str(dtype)})
        self.device, self.dtype, self.metal = torch.device(device), dtype, metal
        for name, value in state.items():
            setattr(self, name, value)
        if "reward_circuit" in self.__dict__:
            self.reward_circuit.apply_weights()

    def execution_summary(self):
        gpu = self.voltage.device.type == 'mps'
        if not gpu:
            note = 'CPU float64 reference. Prior rounding, if any, is not reversible.'
        elif self.metal.state_dtype == torch.float16:
            note = 'Experimental FP16 state, weights and arithmetic; rounding changes firing and is not reference-equivalent. All neurons, edges and ticks retained.'
        elif self.metal.weight_dtype == torch.float16:
            note = 'Experimental FP16 weight storage, FP32 state and sums; weight rounding may change spikes. Offline benchmark only.'
        else:
            note = 'MPS float32 may change spike timing; not equivalent to the float64 reference. All neurons, edges and ticks retained.'
        coarse = self.config.timing_rounding == 'ceil'
        if coarse:
            note += ' Coarse neural clock: delays and refractory durations round up; spike timing and behavior change.'
        return {'device': self.voltage.device.type, 'precision': str(self.voltage.dtype),
                'neural_dt_ms': self.config.dt_ms, 'timing_rounding': self.config.timing_rounding,
                'effective_delay_ms': self.delay_steps * self.config.dt_ms,
                'effective_refractory_ms': self.config.integration_ticks(self.config.refractory_ms) * self.config.dt_ms,
                'kernel': self.metal.version if gpu else 'cpu-event-f64-v1',
                'weight_precision': str(self.metal.weight_dtype) if gpu else str(self.weights.dtype),
                'experimental': gpu or coarse, 'history': self.execution_history, 'note': note}

    def set_timestep(self, dt_ms):
        """Change the clock at a shared boundary, retaining state and queued signals.

        Pending events and refractory release times move to the next new tick,
        never an earlier one. Merging pending signals may change floating sums.
        """
        if dt_ms not in {.1, 1., 1000 / 960} or self.config.profile != 'shiu-2024':
            raise ValueError('Supported paper timesteps: 0.1, 1 or 1000/960 ms')
        if dt_ms == self.config.dt_ms:
            return
        old = self.config
        new = replace(old, dt_ms=dt_ms, timing_rounding='exact' if dt_ms == .1 else 'ceil')
        origin = self.time_origin_ms
        tick = round((self.time_ms - origin) / dt_ms)
        if not math.isclose(origin + tick * dt_ms, self.time_ms, abs_tol=1e-8):
            origin, tick = self.time_ms, 0
        # Map on CPU in float64, even when neural state uses half precision.
        last, refractory = self.last_spike_tick.cpu(), self.refractory_ticks.cpu()
        release = tick + torch.ceil((last + refractory - self.tick_index).double() * old.dt_ms / dt_ms - 1e-9).long()
        new_ref = torch.ceil(refractory.double() * old.dt_ms / dt_ms - 1e-9).long()
        new_ref[refractory == old.integration_ticks(old.refractory_ms)] = new.integration_ticks(new.refractory_ms)
        offsets = [math.ceil(i * old.dt_ms / dt_ms - 1e-9) for i in range(len(self.delay_queue))]
        delay_steps = new.integration_ticks(new.delay_ms)
        queue = torch.zeros((max(delay_steps, max(offsets)) + 1, len(self.ids)), dtype=self.dtype)
        previous = self.delay_queue.cpu()
        for offset, target in enumerate(offsets):
            queue[(tick + target) % len(queue)] += previous[(self.tick_index + offset) % len(previous)]
        if not torch.isfinite(queue).all():
            raise ValueError('Coarsening pending signals overflowed; timestep unchanged')
        new_last = (release - new_ref).to(self.device)
        queue, new_ref = queue.to(self.device), new_ref.to(self.device)
        if self.device.type == 'mps':
            torch.mps.synchronize()
        self.execution_history.append({'kind': 'timestep', 'from_dt_ms': old.dt_ms,
                                       'to_dt_ms': dt_ms, 'time_ms': self.time_ms})
        self.config, self.delay_steps, self.tick_index = new, delay_steps, tick
        self.time_origin_ms = origin
        self.delay_queue, self.refractory_ticks, self.last_spike_tick = queue, new_ref, new_last

    def reset_paper(self, seed=0):
        n = len(self.ids)
        self.voltage = torch.full((n,), -52., dtype=self.dtype)
        self.current = torch.zeros(n, dtype=self.dtype)
        self.spikes = torch.zeros(n, dtype=self.dtype)
        self.rates = torch.zeros(n, dtype=self.dtype)
        self.output_gain = torch.ones(n, dtype=self.dtype)
        self.last_spike_tick = torch.full((n,), -10**12, dtype=torch.int64)
        self.refractory_ticks = torch.full((n,), self.config.integration_ticks(self.config.refractory_ms), dtype=torch.int64)
        self.delay_queue = torch.zeros((self.delay_steps + 1, n), dtype=self.dtype)
        self.spike_counts = torch.zeros(n, dtype=torch.int64)
        self.generator = torch.Generator().manual_seed(seed)
        self.tick_index = 0
        self.time_origin_ms = 0.
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
        self.time_ms = self.time_origin_ms + self.tick_index * c.dt_ms
        self.last_wall_seconds = time.perf_counter() - started
        return self.rates

    def _advance_metal(self, drive, steps, silence, voltage_events, poisson_hz, inputs):
        """Keep all neural state on GPU; transfer external stimuli per batch."""
        c = self.config
        started = time.perf_counter()
        a, b = math.exp(-c.dt_ms / c.membrane_ms), math.exp(-c.dt_ms / c.synapse_ms)
        coupling = c.dt_ms / c.membrane_ms * a if c.synapse_ms == c.membrane_ms else c.synapse_ms / (c.membrane_ms - c.synapse_ms) * (a - b)
        decay = math.exp(-c.dt_ms / 50.)
        parameters = torch.tensor([a, b, coupling, 1-a, decay, (1-decay)*1000/c.dt_ms], device=self.device, dtype=self.dtype)
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
        self.time_ms = self.time_origin_ms + self.tick_index*c.dt_ms
        self.last_wall_seconds = time.perf_counter() - started
        return self.rates
