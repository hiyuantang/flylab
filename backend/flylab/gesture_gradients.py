"""Full-graph surrogate-gradient training. Forward uses the paper's discrete LIF.

Backward uses a fast-sigmoid spike derivative and stops gradients at discrete
refractory/reset decisions. No connectivity or forward neural ticks are removed.
Sparse delivery has a custom VJP to avoid materializing a dense N x N gradient.
"""
import math
import numpy as np
import torch
from .motor_mapping import LEGACY_PROFILE


# Calibrated on the full 500 ms circuit; this scales backward derivatives only.
DEFAULT_SURROGATE_SCALE = .0003


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, scale=1.):
        ctx.scale = scale
        ctx.save_for_backward(voltage)
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        result = gradient * ctx.scale / (1 + voltage.abs()).square()
        return (result, None) if len(ctx.needs_input_grad) == 2 else result


class Delivery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, emitted, factors, engine):
        ctx.engine = engine
        ctx.save_for_backward(emitted)
        return engine.brain.deliver(emitted)

    @staticmethod
    def backward(ctx, gradient):
        engine = ctx.engine
        emitted, = ctx.saved_tensors
        # All existing connections participate in the state derivative, including
        # connections whose presynaptic neuron did not fire in this forward tick.
        grad_emitted = torch.mv(engine.outgoing, gradient)
        grad_factors = torch.zeros(engine.adapter.group_count ** 2, dtype=gradient.dtype)
        active = emitted.nonzero().flatten()
        ptr = engine.brain.out_ptr
        starts, lengths = ptr[active], ptr[active + 1] - ptr[active]
        total = int(lengths.sum())
        if total:
            offsets = torch.repeat_interleave(starts - (lengths.cumsum(0) - lengths), lengths)
            edges = offsets + torch.arange(total)
            amplitudes = torch.repeat_interleave(emitted[active], lengths)
            values = gradient[engine.brain.out_post[edges]] * engine.adapter.base_out[edges] * amplitudes
            grad_factors.index_add_(0, engine.out_pairs[edges], values)
        return grad_emitted, grad_factors.reshape(engine.adapter.group_count, -1), None


class GradientBrain:
    def __init__(self, brain, adapter, bridge, surrogate_scale=1.):
        if brain.device.type != 'cpu' or brain.dtype != torch.float64:
            raise ValueError('Gradient training requires explicit CPU float64 execution')
        if not math.isfinite(surrogate_scale) or not 0 < surrogate_scale <= 1:
            raise ValueError("Surrogate scale must be in (0, 1]")
        self.surrogate_scale = surrogate_scale
        self.brain, self.adapter, self.bridge = brain, adapter, bridge
        self.parameters = torch.nn.Parameter(torch.from_numpy(adapter.initial.copy()))
        self.out_pairs = torch.from_numpy(adapter.out_pairs.astype(np.int64))
        self.outgoing = torch.sparse_csr_tensor(brain.out_ptr, brain.out_post, brain.out_weight,
                                              size=brain.weights.shape, check_invariants=True)
        self.queue = list(brain.delay_queue.unbind())

    def sync_weights(self):
        self.adapter.apply(self.parameters.detach().numpy())

    def reset(self):
        self.queue = list(self.brain.delay_queue.unbind())

    def detach_state(self):
        for name in ('voltage', 'current', 'rates', 'spikes'):
            setattr(self.brain, name, getattr(self.brain, name).detach())
        self.queue = [x.detach() for x in self.queue]
        self.brain.delay_queue = torch.stack(self.queue)

    def advance(self, drive, duration_ms=20., silence=None):
        brain, c = self.brain, self.brain.config
        ticks = round(duration_ms / c.dt_ms)
        if ticks < 1 or not math.isclose(ticks * c.dt_ms, duration_ms):
            raise ValueError('Duration must cover complete neural ticks')
        u, v = self.parameters.unbind()
        factors = 1 + .75 * torch.tanh(u @ v.T / math.sqrt(self.adapter.rank))
        a, b = math.exp(-c.dt_ms / c.membrane_ms), math.exp(-c.dt_ms / c.synapse_ms)
        coupling = c.dt_ms / c.membrane_ms * a if c.synapse_ms == c.membrane_ms else c.synapse_ms / (c.membrane_ms - c.synapse_ms) * (a - b)
        decay = math.exp(-c.dt_ms / 50.)
        drive = drive.to(torch.float64)
        for _ in range(ticks):
            free = brain.tick_index - brain.last_spike_tick >= brain.refractory_ticks
            voltage = -52. + (brain.voltage + 52.) * a + brain.current * coupling + drive * (1 - a)
            voltage = torch.where(free, voltage, brain.voltage)
            current = torch.where(free, brain.current * b, brain.current)
            if silence is not None:
                voltage = torch.where(silence, -52., voltage)
            spikes = Spike.apply(voltage + 45., self.surrogate_scale) * free
            fired = spikes.detach().bool()
            brain.last_spike_tick[fired] = brain.tick_index
            receiving = free & ~fired
            slot = brain.tick_index % len(self.queue)
            target = (brain.tick_index + brain.delay_steps) % len(self.queue)
            self.queue[target] = self.queue[target] + spikes * brain.output_gain
            incoming = Delivery.apply(self.queue[slot], factors, self)
            self.queue[slot] = torch.zeros_like(spikes)
            brain.voltage = torch.where(fired, -52., voltage)
            brain.current = torch.where(fired, 0., current + incoming * receiving)
            brain.spikes = spikes
            brain.spike_counts.add_(fired)
            brain.rates = brain.rates * decay + spikes * ((1 - decay) * 1000 / c.dt_ms)
            brain.tick_index += 1
        brain.total_spikes = int(brain.spike_counts.sum())
        brain.time_ms = brain.time_origin_ms + brain.tick_index * c.dt_ms
        return brain.rates

    def muscles(self):
        rates = self.brain.rates
        action = []
        for ids, weights in zip(self.bridge.channels, self.bridge.channel_weights):
            rate = ((rates[ids] if self.bridge.mapping_profile == LEGACY_PROFILE else rates[ids] * weights).mean()
                    if len(ids) else rates.sum() * 0)
            action.append((rate / 100 * float(np.exp(self.bridge.parameters[7]))).clamp(0, 1))
        # Match the existing bridge's float32 output rounding; gradient stays live.
        return torch.stack(action).float().double()


def muscle_activation(action, previous, model, duration=.02):
    """PyTorch equivalent of MuJoCo's unsmoothed muscle activation Euler update.

    Validated separately against mju_muscleDynamics and the body's actual act.
    This differentiates activation dynamics only, not force or rigid-body physics.
    """
    import mujoco
    if np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE) or np.any(model.actuator_dynprm[:, 2] != 0):
        raise ValueError('Training requires unsmoothed muscle activation dynamics')
    if model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST, mujoco.mjtIntegrator.mjINT_IMPLICIT):
        raise ValueError('Unsupported muscle activation integrator')
    if model.na != model.nu or not np.array_equal(model.actuator_actadr, np.arange(model.nu)):
        raise ValueError('Training requires one activation state per actuator')
    params = torch.as_tensor(model.actuator_dynprm[:, :2].copy(), device=action.device, dtype=action.dtype)
    state = previous
    steps = math.ceil(duration / model.opt.timestep)
    dt = duration / steps
    for _ in range(steps):
        scale = .5 + 1.5 * state.clamp(0, 1)
        delta = action.clamp(0, 1) - state
        tau = torch.where(delta > 0, params[:, 0] * scale, params[:, 1] / scale)
        state = state + dt * delta / tau.clamp_min(1e-15)
    return state
