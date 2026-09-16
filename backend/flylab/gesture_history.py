"""Checkpointed full-sample neural BPTT with explicit boundary-state adjoints.

Only neural state is differentiable here. Ray sensing and MuJoCo kinematics
remain detached; muscle activation dynamics are differentiated by the caller.
Recompute each 20 ms block during backward, retaining only boundary states.
"""
from types import SimpleNamespace
import torch


class FullHistoryWindow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factors, voltage, current, rates, queue, engine, drive, silence, ticks):
        from .gesture_metal import NeuralWindow
        from .gesture_parallel import BatchWindow
        batch = hasattr(engine, 'shared')
        brain = engine if batch else engine.brain
        ctx.engine, ctx.batch, ctx.start, ctx.ticks = engine, batch, brain.tick_index, ticks
        ctx.save_for_backward(voltage, current, rates, queue, drive, silence,
                              brain.last_spike_tick.clone(), brain.spike_counts.clone(),
                              brain.output_gain.clone())
        # The ordinary forward follows the existing FP16 kernel exactly. Its
        # per-tick tape is discarded; backward reconstructs one block at a time.
        temporary = SimpleNamespace(save_for_backward=lambda *values: None, record_history=False, reuse_buffers=True)
        (BatchWindow if batch else NeuralWindow).forward(temporary, factors, engine, drive, silence, ticks)
        ctx.set_materialize_grads(False)
        if getattr(ctx, 'rates_only', False):
            return brain.rates.float().clone(), None, None, None
        return tuple(value.float().clone() for value in
                     (brain.rates, brain.voltage, brain.current, brain.delay_queue))

    @staticmethod
    def backward(ctx, rates_grad, voltage_grad, current_grad, queue_grad):
        from .gesture_metal import NeuralWindow, reverse_neural_window
        from .gesture_parallel import BatchWindow, reverse_batch_window
        engine = ctx.engine
        brain = engine if ctx.batch else engine.brain
        v, g, r, q, drive, silence, last, counts, gain = ctx.saved_tensors
        names = ('voltage', 'current', 'rates', 'delay_queue', 'last_spike_tick',
                 'spike_counts', 'output_gain', 'spikes')
        original = {name: getattr(brain, name) for name in names}
        original_tick = brain.tick_index
        original_time = getattr(brain, 'time_ms', None)
        try:
            for name, value in zip(names[:-1], (v, g, r, q, last, counts, gain)):
                setattr(brain, name, value.detach().to(dtype=original[name].dtype).clone())
            brain.spikes = torch.zeros_like(brain.voltage)
            brain.tick_index = ctx.start
            tape = SimpleNamespace(reuse_buffers=True)
            tape.save_for_backward = lambda *values: setattr(tape, 'saved_tensors', values)
            window = BatchWindow if ctx.batch else NeuralWindow
            window.forward(tape, None, engine, drive, silence, ctx.ticks)
            zero = lambda grad, state: torch.zeros_like(state, dtype=torch.float32) if grad is None else grad.contiguous()
            reverse = reverse_batch_window if ctx.batch else reverse_neural_window
            df, dv, dg, dr, dq = reverse(tape, zero(rates_grad, r), zero(voltage_grad, v),
                                         zero(current_grad, g), zero(queue_grad, q))
            return df, dv, dg, dr, dq, None, None, None, None
        finally:
            for name, value in original.items():
                setattr(brain, name, value)
            brain.tick_index = original_tick
            if original_time is not None:
                brain.time_ms = original_time


def advance_full_history(engine, factors, drive, silence, ticks):
    brain = engine if hasattr(engine, 'shared') else engine.brain
    shared = engine.shared if hasattr(engine, 'shared') else engine
    if getattr(shared, 'stable_history', False):
        if not hasattr(engine, 'history_sequence'):
            engine.history_sequence = []
        state = tuple(value.detach().clone() for value in
                      (brain.voltage, brain.current, brain.rates, brain.delay_queue))
        ctx = SimpleNamespace(set_materialize_grads=lambda value: None, rates_only=True)
        ctx.save_for_backward = lambda *values: setattr(ctx, 'saved_tensors', values)
        with torch.no_grad():
            rates, *_ = FullHistoryWindow.forward(ctx, factors, *state, engine, drive, silence, ticks)
        rates = rates.detach().requires_grad_()
        engine.history_sequence.append((ctx, rates))
        return rates
    state = getattr(engine, 'history_state', None)
    if state is None:
        state = tuple(value.detach().float().clone() for value in
                      (brain.voltage, brain.current, brain.rates, brain.delay_queue))
    rates, voltage, current, queue = FullHistoryWindow.apply(factors, *state, engine, drive, silence, ticks)
    engine.history_state = (voltage, current, rates, queue)
    return rates


def check_history_memory(engine, steps):
    brain = engine if hasattr(engine, 'shared') else engine.brain
    n = brain.voltage.numel()
    # Exact native FP16 boundary state plus drive, mask and reset history.
    boundary = n * (6 + 2 * len(brain.delay_queue) + 21)
    scratch = n * (200 * 8 + 160)
    required = steps * boundary + scratch
    available = .9 * torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()
    if required > available:
        raise ValueError(f'Full-history training needs about {required / 1e9:.1f} GB additional GPU memory; reduce batch size or sample duration')


def window_buffers(engine, shape, reuse=False):
    """Reuse bounded scratch only when no autograd tape retains its contents."""
    key = tuple(shape)
    if reuse:
        if not hasattr(engine, '_window_buffers'):
            engine._window_buffers = {}
        if key in engine._window_buffers:
            return engine._window_buffers[key]
    values = (torch.empty(shape, device='mps', dtype=torch.float16),
              torch.empty(shape, device='mps', dtype=torch.int32),
              torch.empty(shape, device='mps', dtype=torch.float16))
    if reuse:
        engine._window_buffers[key] = values
    return values


def backward_full_history(engine, loss, stop=None, progress=None):
    """Block-floating VJP: one common power-of-two exponent, no state clipping.

    Muscle-history autograd supplies each command's local rate adjoint. Neural
    adjoints and accumulated factor derivatives share a scale, so rescaling
    preserves their relative contributions before ordinary global norm clipping.
    """
    import math
    sequence = engine.history_sequence
    local = torch.autograd.grad(loss, [rates for _, rates in sequence], allow_unused=True)
    shared = engine.shared if hasattr(engine, 'shared') else engine
    shape = sequence[-1][0].saved_tensors[0].shape
    dv = torch.zeros(shape, device='mps')
    dg, dr = torch.zeros_like(dv), torch.zeros_like(dv)
    dq = torch.zeros_like(sequence[-1][0].saved_tensors[3], dtype=torch.float32)
    df = torch.zeros((shared.adapter.group_count, shared.adapter.group_count), device='mps')
    exponent = 0
    for completed, ((ctx, _), gradient) in enumerate(zip(reversed(sequence), reversed(local)), 1):
        if stop is not None and stop.is_set():
            from .gesture_training import Cancelled
            raise Cancelled()
        # Align each local loss contribution with the shared binary exponent.
        addition = torch.zeros_like(dr) if gradient is None else scale_power_two(gradient, -exponent)
        result = FullHistoryWindow.backward(ctx, dr + addition, dv, dg, dq)
        contribution, dv, dg, dr, dq = result[:5]
        df = df + contribution
        maximum = float(torch.stack([value.abs().max() for value in (df, dv, dg, dr, dq)]).max())
        if not math.isfinite(maximum):
            raise ValueError('Nonfinite adjoint within a checkpoint block')
        if maximum:
            shift = math.frexp(maximum)[1]
            factor = math.ldexp(1., -shift)
            df, dv, dg, dr, dq = (value * factor for value in (df, dv, dg, dr, dq))
            exponent += shift
        if progress:
            progress(backward_step=completed, backward_total=len(sequence))
    u, v = shared.parameters.unbind()
    factors = 1 + .75 * torch.tanh(u @ v.T / math.sqrt(shared.adapter.rank))
    parameter_gradient, = torch.autograd.grad(factors, shared.parameters, df)
    if shared.parameters.grad is not None:
        previous_exponent = getattr(shared, 'gradient_exponent', 0)
        common_exponent = max(exponent, previous_exponent)
        parameter_gradient = (scale_power_two(parameter_gradient, exponent - common_exponent)
                              + scale_power_two(shared.parameters.grad, previous_exponent - common_exponent))
        exponent = common_exponent
    shared.parameters.grad = parameter_gradient
    shared.gradient_exponent = exponent
    engine.history_sequence = []


def clip_full_gradient(engine):
    """Apply the existing unit global-norm limit without overflowing its norm."""
    import math
    gradient = engine.parameters.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite or missing adapter gradient')
    maximum = float(gradient.abs().max())
    if not maximum:
        engine.gradient_exponent = 0
        return 0., None
    unit = gradient / maximum
    unit_norm = float(torch.linalg.vector_norm(unit))
    exponent = getattr(engine, 'gradient_exponent', 0)
    log_norm = math.log10(maximum) + math.log10(unit_norm) + exponent * math.log10(2.)
    if log_norm > 0:
        gradient.copy_(unit / unit_norm)
    elif exponent:
        gradient.copy_(scale_power_two(gradient, exponent))
    engine.gradient_exponent = 0
    return (10. ** log_norm if log_norm < 308 else None), log_norm


def scale_power_two(value, exponent):
    """Scale without first overflowing/underflowing a scalar FP32 multiplier."""
    while exponent > 64:
        value = value * (2. ** 64)
        exponent -= 64
    while exponent < -64:
        value = value * (2. ** -64)
        exponent += 64
    return value * (2. ** exponent)
