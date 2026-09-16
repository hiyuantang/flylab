import numpy as np
import pytest
import torch
from dataclasses import replace
from test_paper_dynamics import TinyBrain
from test_full_connectome import graph_fixture
from flylab.simulation import Simulation
from flylab.live_state import load_live, save_live, NEURAL_TENSORS
from flylab.metal_dynamics import ActiveRowMetalDynamics

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')


def test_precision_switch_preserves_all_state_and_rejects_overflow():
    brain = TinyBrain([[0, 20], [10, 0]])
    brain.set_device('mps')
    brain.advance_paper(torch.tensor([25., 0.]), 20.)
    before = {k: getattr(brain, k).cpu().clone() for k in NEURAL_TENSORS}
    rng, tick, time = brain.generator.get_state(), brain.tick_index, brain.time_ms
    brain.set_device('mps', 'float16')
    assert isinstance(brain.metal, ActiveRowMetalDynamics) and brain.metal.mark_lanes == 4
    for key, value in before.items():
        torch.testing.assert_close(getattr(brain, key).cpu(), value.half() if value.is_floating_point() else value, rtol=0, atol=0)
    assert torch.equal(rng, brain.generator.get_state())
    assert (tick, time) == (brain.tick_index, brain.time_ms)
    engine = brain.metal
    brain.set_device('mps')
    assert brain.metal is engine
    brain.reset_paper()
    assert brain.voltage.dtype == torch.float16 and brain.metal is engine
    brain.set_device('mps', 'float32')
    brain.current[0] = 1e6
    with pytest.raises(ValueError, match='overflowed current'):
        brain.set_device('mps', 'float16')
    assert brain.dtype == torch.float32 and brain.current[0] == 1e6
    with pytest.raises(ValueError, match='CPU requires'):
        brain.set_device('cpu', 'float16')


@pytest.mark.parametrize('return_to', [None, 'float32', 'float64'])
def test_half_checkpoint_exact_continuation_and_reconfiguration(tmp_path, return_to):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', device='mps')
    sim.bridge.set_parameters(np.array([.3, -.7, .9, .1, -.2, .23, .33, .5]))
    sim.advance()
    sim.set_command_hz(60)
    sim.full_brain.set_device('mps', 'float16')
    sim.advance(2)
    if return_to:
        sim.full_brain.set_device('cpu' if return_to == 'float64' else 'mps', return_to)
    path = tmp_path / 'precision.pt'
    save_live(sim, path)
    restored = load_live(path, tmp_path / 'full')
    assert restored.command_hz == 60 and restored.time == sim.time
    assert restored.full_brain.execution_summary() == sim.full_brain.execution_summary()
    sim.advance(2)
    restored.advance(2)
    for key in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, key), getattr(restored.full_brain, key), rtol=0, atol=0)
    np.testing.assert_array_equal(sim.body.data.qpos, restored.body.data.qpos)
    np.testing.assert_array_equal(sim.body.data.qvel, restored.body.data.qvel)
    dtype = sim.full_brain.dtype
    sim.configure('connectome', physiology=replace(sim.full_brain.config, gain=.3))
    assert sim.full_brain.dtype == dtype
    sim.configure_scene('lab')
    assert sim.full_brain.dtype == dtype
    sim.reset()
    assert sim.full_brain.dtype == dtype
