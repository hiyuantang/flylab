import numpy as np
import pytest
import torch
from test_paper_dynamics import TinyBrain
from flylab.metal_dynamics import MetalDynamics, ActiveRowMetalDynamics, FusedActiveRowMetalDynamics, NEURAL_TENSORS

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')


def make_brain(weights, active=False, **parameters):
    brain = TinyBrain(weights, **parameters)
    brain.set_device('mps')
    if active == 'fused':
        brain.metal = FusedActiveRowMetalDynamics(brain.weights)
    elif isinstance(active, str) and active.startswith('mark'):
        brain.metal = ActiveRowMetalDynamics(brain.weights, mark_lanes=int(active[4:]))
    elif active:
        brain.metal = ActiveRowMetalDynamics(brain.weights)
    else:
        brain.metal = MetalDynamics(brain.weights, weight_dtype=torch.float16, state_dtype=torch.float16)
    brain.dtype = torch.float16
    brain.reset_paper()
    return brain


@pytest.mark.parametrize('candidate', [True, 'fused', 'mark1', 'mark8', 'mark32'])
def test_active_rows_match_every_sum_and_clear_stale_flags(candidate):
    rng = np.random.default_rng(31)
    w = rng.normal(size=(257, 257))
    w[np.abs(w) < .25] = 0
    w[17] = 0
    brains = [make_brain(w, active=mode) for mode in [False, candidate]]
    for step in range(8):
        values = np.zeros(257)
        if step % 3:
            values[::(step+1)] = rng.uniform(-2, 2, len(values[::(step+1)]))
        emitted = torch.tensor(values, device='mps', dtype=torch.float16)
        outputs = [b.deliver(emitted).clone() for b in brains]
        torch.testing.assert_close(*outputs, atol=0, rtol=0)
    engine = brains[1].metal
    assert torch.equal(engine.pre.cpu().long(), brains[1].weights.col_indices())
    assert torch.equal(engine.ptr.cpu().long(), brains[1].weights.crow_indices())
    assert engine.out_post.numel() == engine.pre.numel()


@pytest.mark.parametrize('delay_ms', [0., 1.8])
@pytest.mark.parametrize('candidate', [True, 'fused', 'mark1', 'mark8', 'mark32'])
def test_active_full_state_matches_baseline_each_tick(delay_ms, candidate):
    w = [[0, 1, -2, 0], [50, 0, 0, 0], [0, 30, 0, 0], [2, 0, 0, 0]]
    brains = [make_brain(w, active=mode, delay_ms=delay_ms) for mode in [False, candidate]]
    for b in brains:
        b.output_gain[1] = .3
    events = torch.zeros((100, 4), dtype=torch.float16)
    events[[0, 30, 60], 0] = 68.75
    for i in range(len(events)):
        for b in brains:
            b.advance_paper(torch.zeros(4), .1, voltage_events=events[i:i+1],
                            silence=torch.tensor([False, False, False, True]))
        for name in NEURAL_TENSORS:
            torch.testing.assert_close(getattr(brains[0], name), getattr(brains[1], name), atol=0, rtol=0)
    assert brains[1].total_spikes > 0


@pytest.mark.parametrize('candidate', [True, 'fused', 'mark1', 'mark8', 'mark32'])
def test_active_poisson_batching_and_rng_are_unchanged(candidate):
    w = [[0, 1, -2], [50, 0, 0], [0, 30, 0]]
    brains = [make_brain(w, active=mode) for mode in [False, candidate, candidate]]
    for b in brains:
        b.refractory_ticks[0] = 0
    for b in brains[:2]:
        b.advance_paper(torch.zeros(3), 20., poisson_hz=torch.tensor([200., 0., 0.]))
    for _ in range(20):
        brains[2].advance_paper(torch.zeros(3), 1., poisson_hz=torch.tensor([200., 0., 0.]))
    for b in brains[1:]:
        for name in NEURAL_TENSORS:
            torch.testing.assert_close(getattr(brains[0], name), getattr(b, name), atol=0, rtol=0)
        assert torch.equal(b.generator.get_state(), brains[0].generator.get_state())
