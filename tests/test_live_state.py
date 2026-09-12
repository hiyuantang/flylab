import numpy as np
import pytest
import torch
from test_full_connectome import graph_fixture
from flylab.full_connectome import Physiology
from flylab.simulation import Simulation
from flylab.live_state import save_live, load_live, NEURAL_TENSORS


def test_full_live_continuation_preserves_neural_delay_body_and_clocks(tmp_path):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', Physiology.paper())
    sim.stimulate('antennal', 3, .5)
    sim.advance()
    path = tmp_path / 'live.pt'
    saved = save_live(sim, path)
    assert saved['neurons'] == 3 and saved['time'] == .02
    restored = load_live(path, tmp_path / 'full')
    assert not restored.running and restored.pulses == sim.pulses
    assert restored.time == sim.time and restored.wall_seconds == sim.wall_seconds
    sim.advance(3)
    restored.advance(3)
    for key in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, key), getattr(restored.full_brain, key), rtol=0, atol=0)
    for key in ['qpos', 'qvel', 'act', 'qacc_warmstart']:
        np.testing.assert_array_equal(getattr(sim.body.data, key), getattr(restored.body.data, key))
    assert sim.history == restored.history
    assert restored.snapshot()['timing']['simulated_seconds'] == .08


def test_checkpoint_rejects_neuron_loss_or_clock_mismatch(tmp_path):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', Physiology.paper())
    path = tmp_path / 'live.pt'
    save_live(sim, path)
    p = torch.load(path, weights_only=True)
    p['neural']['voltage'] = p['neural']['voltage'][:2]
    torch.save(p, path)
    with pytest.raises(ValueError, match='complete neural state'):
        load_live(path, tmp_path / 'full')
    assert sim.steps == 0


def test_mapping_profile_round_trip_and_legacy_default(tmp_path):
    from flylab.motor_mapping import MAPPING_PROFILE, LEGACY_PROFILE
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', Physiology.paper())
    path = tmp_path / 'live.pt'
    save_live(sim, path)
    assert load_live(path, tmp_path / 'full').bridge.mapping_profile == MAPPING_PROFILE
    p = torch.load(path, weights_only=True)
    del p['mapping_profile']
    torch.save(p, path)
    assert load_live(path, tmp_path / 'full').bridge.mapping_profile == LEGACY_PROFILE
    p['mapping_profile'] = 'unknown'
    torch.save(p, path)
    with pytest.raises(ValueError, match='mapping profile'):
        load_live(path, tmp_path / 'full')
