import numpy as np
import pytest
import torch
from test_paper_dynamics import TinyBrain
from test_full_connectome import graph_fixture
from flylab.live_state import load_live, save_live, NEURAL_TENSORS
from flylab.simulation import Simulation


gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')


def test_unavailable_device_does_not_mutate_state(monkeypatch):
    brain = TinyBrain([[0, 1], [1, 0]])
    before = brain.voltage.clone()
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='unavailable'):
        brain.set_device('mps')
    assert brain.device.type == 'cpu' and brain.execution_history == []
    torch.testing.assert_close(brain.voltage, before, rtol=0, atol=0)


@gpu
def test_metal_csr_covers_inhibition_disconnected_and_high_degree_rows():
    rng = np.random.default_rng(17)
    weights = rng.normal(size=(257, 257))
    weights[weights < -.5] = 0
    weights[13] = 0
    weights[12] = rng.normal(size=257)
    brain = TinyBrain(weights)
    brain.set_device('mps')
    emitted = torch.tensor(rng.uniform(size=257), dtype=torch.float32, device='mps')
    delivered = brain.deliver(emitted).cpu()
    expected = brain.weights @ emitted.cpu().double()
    torch.testing.assert_close(delivered.double(), expected, atol=3e-6, rtol=2e-6)
    assert len(brain.metal.pre) == brain.weights._nnz()
    assert brain.voltage.device.type == 'mps'


@gpu
def test_mps_poisson_batching_clamp_and_migration_preserve_pending_events():
    w = [[0, 1, -2], [50, 0, 0], [0, 30, 0]]
    brains = [TinyBrain(w) for _ in range(3)]
    for brain in brains:
        brain.refractory_ticks[0] = 0
        brain.output_gain[1] = .3
    for brain in brains[1:]:
        brain.set_device('mps')
    kwargs = {'poisson_hz': torch.tensor([200., 0., 0.]), 'silence': torch.tensor([False, False, True])}
    brains[0].advance_paper(torch.zeros(3), 20, **kwargs)
    brains[1].advance_paper(torch.zeros(3), 20, **kwargs)
    for _ in range(20):
        brains[2].advance_paper(torch.zeros(3), 1, **kwargs)
    for key in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(brains[1], key), getattr(brains[2], key), rtol=0, atol=0)
    torch.testing.assert_close(brains[0].spike_counts, brains[1].spike_counts.cpu(), rtol=0, atol=0)
    assert brains[1].spike_counts[2].item() == 0
    assert torch.equal(brains[0].generator.get_state(), brains[1].generator.get_state())
    pending = brains[1].delay_queue.cpu().double()
    brains[1].set_device('cpu')
    torch.testing.assert_close(brains[1].delay_queue, pending, rtol=0, atol=0)
    assert brains[1].tick_index == 200


@gpu
def test_mps_live_checkpoint_resumes_brain_and_physics_exactly(tmp_path):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', device='mps')
    sim.stimulate('antennal', 3, .5)
    sim.advance()
    path = tmp_path / 'live.pt'
    save_live(sim, path)
    payload = torch.load(path, weights_only=True)
    assert payload['execution']['device'] == 'mps'
    assert all(t.device.type == 'cpu' for t in payload['neural'].values())
    restored = load_live(path, tmp_path / 'full')
    assert restored.full_brain.voltage.device.type == 'mps'
    sim.advance(2)
    restored.advance(2)
    for key in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, key), getattr(restored.full_brain, key), atol=0, rtol=0)
    for key in ['qpos', 'qvel', 'act', 'qacc_warmstart']:
        np.testing.assert_array_equal(getattr(sim.body.data, key), getattr(restored.body.data, key))
    payload['execution']['precision'] = 'torch.float64'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='precision or kernel'):
        load_live(path, tmp_path / 'full')


@gpu
def test_half_weights_preserve_edges_and_accumulate_in_float32():
    from flylab.metal_dynamics import MetalDynamics, MIXED_METAL_VERSION
    rng = np.random.default_rng(321)
    weights = rng.normal(size=(257, 257))
    weights[np.abs(weights) < .1] = 0
    weights[13] = 0
    brain = TinyBrain(weights)
    brain.set_device('mps')
    original = brain.weights.values().clone()
    brain.metal = MetalDynamics(brain.weights, weight_dtype=torch.float16)
    emitted = torch.tensor(rng.uniform(size=257), dtype=torch.float32, device='mps')
    actual = brain.deliver(emitted).cpu()
    # Compare to rounded weights with full-precision products/sums, not an FP16 dot.
    expected = torch.tensor(weights, dtype=torch.float32).half().double() @ emitted.cpu().double()
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=2e-6)
    assert torch.equal(brain.weights.values(), original)
    assert torch.equal(brain.metal.pre.cpu().long(), brain.weights.col_indices())
    assert torch.equal(brain.metal.ptr.cpu().long(), brain.weights.crow_indices())
    assert brain.metal.weight.dtype == torch.float16
    assert actual.dtype == torch.float32
    assert brain.execution_summary()['kernel'] == MIXED_METAL_VERSION
    assert brain.execution_summary()['weight_precision'] == 'torch.float16'
    brain.advance_paper(torch.full((257,), 14.7), 5)
    for key in NEURAL_TENSORS:
        tensor = getattr(brain, key)
        assert tensor.device.type == 'mps'
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    assert brain.tick_index == 50


@gpu
@pytest.mark.parametrize('weight,reason', [(70000., 'overflow'), (1e-9, 'underflow')])
def test_half_weights_reject_destructive_conversion(weight, reason):
    from flylab.metal_dynamics import MetalDynamics
    brain = TinyBrain([[0, weight], [0, 0]])
    original = brain.weights.values().clone()
    with pytest.raises(ValueError, match=reason):
        MetalDynamics(brain.weights, weight_dtype=torch.float16)
    assert torch.equal(brain.weights.values(), original)


@gpu
def test_half_weight_schedule_matches_fp32_when_weights_are_exactly_representable():
    from flylab.metal_dynamics import MetalDynamics
    brains = [TinyBrain([[0, 1, -2], [50, 0, 0], [0, 30, 0]]) for _ in range(2)]
    for brain in brains:
        brain.set_device('mps')
    brains[1].metal = MetalDynamics(brains[1].weights, weight_dtype=torch.float16)
    for _ in range(10):
        for brain in brains:
            brain.advance_paper(torch.tensor([14.7, 0, 0]), 2)
        for name in NEURAL_TENSORS:
            torch.testing.assert_close(getattr(brains[0], name), getattr(brains[1], name), atol=0, rtol=0)


@gpu
def test_all_half_uses_half_state_parameters_and_exact_integer_timing():
    from flylab.metal_dynamics import MetalDynamics, HALF_METAL_VERSION
    brain = TinyBrain([[0, 1, -2], [50, 0, 0], [0, 30, 0]])
    brain.set_device('mps')
    brain.metal = MetalDynamics(brain.weights, weight_dtype=torch.float16, state_dtype=torch.float16)
    brain.dtype = torch.float16
    brain.reset_paper()
    emitted = torch.tensor([1., .5, .25], device='mps', dtype=torch.float16)
    expected = (brain.weights @ emitted.cpu().double()).half()
    torch.testing.assert_close(brain.deliver(emitted).cpu(), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match='Signal precision'):
        brain.deliver(emitted.float())
    for _ in range(10):
        brain.advance_paper(torch.tensor([14.7, 0., 0.]), 2.)
    assert brain.tick_index == 200 and brain.time_ms == 20.
    assert brain.total_spikes > 0
    assert brain.execution_summary()['kernel'] == HALF_METAL_VERSION
    assert brain.execution_summary()['precision'] == 'torch.float16'
    for name in NEURAL_TENSORS:
        tensor = getattr(brain, name)
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float16 and torch.isfinite(tensor).all()
        else:
            assert tensor.dtype == torch.int64
    assert brain.metal.ptr.dtype == torch.int32
    assert brain.metal.pre.dtype == torch.int32


@gpu
def test_all_half_refractory_delay_clamp_and_batching():
    from flylab.metal_dynamics import MetalDynamics
    brains = [TinyBrain([[0, 0, 0], [50, 0, 0], [0, 30, 0]]) for _ in range(2)]
    for brain in brains:
        brain.set_device('mps')
        brain.metal = MetalDynamics(brain.weights, weight_dtype=torch.float16, state_dtype=torch.float16)
        brain.dtype = torch.float16
        brain.reset_paper()
    events = torch.zeros((100, 3), dtype=torch.float16)
    events[[0, 30], 0] = 68.75
    events[0, 2] = 7.  # Equality is not above threshold; neuron is also clamped.
    silence = torch.tensor([False, False, True])
    brains[0].advance_paper(torch.zeros(3), 10., voltage_events=events, silence=silence)
    for step in range(100):
        brains[1].advance_paper(torch.zeros(3), .1, voltage_events=events[step:step+1], silence=silence)
    for name in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(brains[0], name), getattr(brains[1], name), atol=0, rtol=0)
    assert brains[0].spike_counts[0].item() == 2
    assert brains[0].spike_counts[1].item() > 0
    assert brains[0].spike_counts[2].item() == 0
