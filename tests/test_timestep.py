import numpy as np
import pytest
import torch
from test_paper_dynamics import TinyBrain
from test_full_connectome import graph_fixture
from flylab.full_connectome import Physiology
from flylab.simulation import Simulation
from flylab.live_state import load_live, save_live, NEURAL_TENSORS


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_clock_switch_preserves_pending_events_and_refractory_release(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable')
    brain = TinyBrain([[0, 1], [1, 0]])
    brain.set_device(device)
    brain.advance_paper(torch.zeros(2), 20.)
    brain.last_spike_tick[:] = 199  # Release at 22.1 ms: first coarse release is 23 ms.
    brain.delay_queue[(200 + 1) % 19, 0] = 2
    brain.delay_queue[(200 + 9) % 19, 0] = 3
    brain.delay_queue[(200 + 18) % 19, 1] = 4
    before = {k: getattr(brain, k).clone() for k in ['voltage', 'current', 'rates', 'spike_counts', 'output_gain']}
    rng = brain.generator.get_state()
    brain.set_timestep(1.)
    assert brain.tick_index == 20 and brain.time_ms == 20
    assert brain.delay_steps == 2
    assert brain.refractory_ticks.tolist() == [3, 3]
    assert (brain.last_spike_tick + brain.refractory_ticks).tolist() == [23, 23]
    assert brain.delay_queue[(20 + 1) % 3, 0] == 5
    assert brain.delay_queue[(20 + 2) % 3, 1] == 4
    for name, state in before.items():
        torch.testing.assert_close(getattr(brain, name), state, rtol=0, atol=0)
    assert torch.equal(rng, brain.generator.get_state())
    brain.set_timestep(.1)
    assert len(brain.delay_queue) == 21  # Preserve the signal due at 22 ms.
    assert brain.delay_queue[(200 + 20) % 21, 1] == 4
    assert brain.refractory_ticks.tolist() == [22, 22]


def test_timestep_validation_is_explicit_and_non_mutating():
    with pytest.raises(ValueError, match='whole integration ticks'):
        Physiology.paper(dt_ms=1.)
    config = Physiology.paper(dt_ms=1., timing_rounding='ceil')
    assert config.integration_ticks(1.8) == 2
    assert config.integration_ticks(2.2) == 3
    brain = TinyBrain([[0]])
    brain.advance_paper(torch.zeros(1), .1)
    brain.set_timestep(1.)
    assert brain.config.dt_ms == 1 and brain.tick_index == 0
    assert brain.time_ms == brain.time_origin_ms == .1
    with pytest.raises(ValueError, match='Supported paper timesteps'):
        brain.set_timestep(2.)


@pytest.mark.parametrize('refine', [False, True])
def test_coarse_and_refined_checkpoint_continuation(tmp_path, refine):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full')
    sim.stimulate('antennal', 3, .5)
    sim.advance()
    sim.full_brain.set_timestep(1.)
    sim.advance()
    if refine:
        sim.full_brain.set_timestep(.1)
    path = tmp_path / 'coarse.pt'
    save_live(sim, path)
    restored = load_live(path, tmp_path / 'full')
    assert restored.full_brain.config == sim.full_brain.config
    sim.advance()
    restored.advance()
    for key in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, key), getattr(restored.full_brain, key), rtol=0, atol=0)
    np.testing.assert_array_equal(sim.body.data.qpos, restored.body.data.qpos)
    assert sim.time == restored.time == .06


def test_coarse_gpu_scheduling_matches_cpu():
    if not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable')
    brains = [TinyBrain([[0, 10], [40, 0]], dt_ms=1., timing_rounding='ceil') for _ in range(2)]
    brains[1].set_device('mps')
    for _ in range(30):
        for brain in brains:
            brain.advance_paper(torch.tensor([30., 0.]), 1.)
        torch.testing.assert_close(brains[0].spike_counts, brains[1].spike_counts.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(brains[0].voltage, brains[1].voltage.cpu().double(), rtol=0, atol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_60hz_continuation_preserves_clocks_and_velocity(tmp_path, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable')
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', device=device)
    sim.advance()  # Migrate at 20 ms, which is not on a 960 Hz global clock.
    before = sim.body.data.qvel.copy()
    sim.set_command_hz(60)
    np.testing.assert_array_equal(sim.body.data.qvel, before)
    assert sim.full_brain.time_origin_ms == 20
    assert sim.full_brain.config.dt_ms == 1000 / 960
    path = tmp_path / '60hz.pt'
    sim.advance(3)
    assert sim.time == pytest.approx(.07)
    assert sim.full_brain.tick_index == 48
    assert sim.body.data.time == pytest.approx(sim.time, abs=1e-10)
    assert sim.body.model.opt.timestep == .0001
    save_live(sim, path)
    restored = load_live(path, tmp_path / 'full')
    assert restored.command_hz == 60
    sim.advance(3)
    restored.advance(3)
    for key in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, key), getattr(restored.full_brain, key), rtol=0, atol=0)
    np.testing.assert_array_equal(sim.body.data.qpos, restored.body.data.qpos)
    np.testing.assert_array_equal(sim.body.data.qvel, restored.body.data.qvel)
    assert sim.time == pytest.approx(.12)
    sim.set_command_hz(50)
    sim.advance()
    assert sim.time == pytest.approx(.14)
    assert sim.full_brain.time_ms == pytest.approx(140)


def test_physics_retains_velocity_and_activation_between_commands():
    from flylab.body import FlyBody
    body = FlyBody()
    body.data.qvel[0] = 2.
    body.step_muscles(np.full(body.model.nu, .1), dt=1/60)
    first = body.data.act.copy()
    assert np.any(first > 0)
    assert body.data.time == pytest.approx(1/60)
    assert np.linalg.norm(body.data.qvel) > 0
    body.step_muscles(np.full(body.model.nu, .1), dt=1/60)
    assert body.data.time == pytest.approx(2/60)
    assert np.isfinite(body.data.qvel).all()
    assert body.model.opt.timestep == .0001
