from pathlib import Path
import json
from dataclasses import asdict
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch
from flylab.full_connectome import build_full, FullBrain, Physiology
from flylab.neuromuscular import NeuromuscularBridge
from flylab.simulation import Simulation
from flylab.body import BodyParameters
from flylab.env import FlyEnv
from flylab.physical_training import rollout
from flylab.physical_policy import TrialEnvironment


def graph_fixture(directory):
    rows = [
        {'bodyId': 10, 'superclass': 'cb_sensory', 'class': 'olfactory', 'type': 'ORN_test', 'rootSide': 'L', 'somaSide': 'L', 'subclass': None},
        {'bodyId': 20, 'superclass': 'vnc_motor', 'class': None, 'type': 'Ti extensor MN', 'rootSide': None, 'somaSide': 'L', 'subclass': 'fl'},
        {'bodyId': 30, 'superclass': 'vnc_motor', 'class': None, 'type': 'unidentified', 'rootSide': None, 'somaSide': 'L', 'subclass': 'fl'},
        {'bodyId': 40, 'superclass': None, 'class': None, 'type': None, 'rootSide': None, 'somaSide': None, 'subclass': None},
    ]
    feather.write_feather(pa.Table.from_pylist(rows), directory / 'body-annotations.feather')
    feather.write_feather(pa.table({'body_pre': [10, 10, 40, 40], 'body_post': [20, 20, 10, 40], 'weight': [3, 4, 5, 6]}), directory / 'connectome-weights.feather')
    feather.write_feather(pa.table({'body': [10, 20, 30], 'consensus_nt': ['acetylcholine', 'glutamate', 'unclear'], 'predicted_nt_confidence': [.9, .8, .2]}), directory / 'body-neurotransmitters.feather')
    return build_full(directory)


def test_full_import_exact_counts_orientation_and_exclusions(tmp_path):
    m = graph_fixture(tmp_path)
    assert (m['neurons'], m['edges'], m['retained_connection_rows'], m['retained_synapses']) == (3, 1, 2, 7)
    assert m['boundary_connection_rows'] == 1 and m['excluded_internal_rows'] == 1
    np.testing.assert_array_equal(np.load(tmp_path / 'full/indptr.npy'), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.load(tmp_path / 'full/indices.npy'), [0])
    np.testing.assert_array_equal(np.load(tmp_path / 'full/counts.npy'), [7])
    with pytest.raises(ValueError):
        build_full(tmp_path)


def test_stateful_spikes_follow_measured_direction_and_reset(tmp_path):
    graph_fixture(tmp_path)
    brain = FullBrain(tmp_path / 'full', Physiology(gain=20))
    zero = torch.zeros(3)
    brain.advance(zero, 40)
    assert brain.total_spikes == 0
    brain.advance(torch.tensor([3., 0., 0.]), 100)
    assert brain.rates[0] > 0 and brain.rates[1] > 0
    assert brain.rates[2] == 0  # No invented pathway to the disconnected neuron.
    v = brain.voltage.clone()
    rates = brain.rates.clone()
    spikes = brain.total_spikes
    brain.reset()
    for _ in range(5):
        brain.advance(torch.tensor([3., 0., 0.]), 20)
    torch.testing.assert_close(brain.voltage, v)
    torch.testing.assert_close(brain.rates, rates, rtol=0, atol=0)
    assert brain.total_spikes == spikes
    brain.reset()
    brain.advance(torch.tensor([0., 3., 0.]), 100)
    assert brain.rates[0] == 0  # Edges are directed, not transposed.
    with pytest.raises(ValueError):
        brain.advance(torch.tensor([float('nan'), 0., 0.]))


def test_peripheral_mapping_never_assigns_unknown_targets(tmp_path):
    graph_fixture(tmp_path)
    brain = FullBrain(tmp_path / 'full')
    bridge = NeuromuscularBridge(brain)
    assert bridge.summary()['mapped_motor_neurons'] == 1
    assert bridge.summary()['unmapped_motor_neurons'] == 1
    brain.rates[1] = 50
    action = bridge.muscles()
    assert action[11] == .5  # LF tibia extension requires negative rig torque.
    assert np.count_nonzero(action) == 1
    original = brain.weights.values().clone()
    bridge.set_parameters(np.ones(8, dtype=np.float32) * .2)
    torch.testing.assert_close(brain.weights.values(), original)


def test_full_dataset_integer_integrity_when_present():
    directory = Path('data/full')
    if not directory.exists():
        pytest.skip('Optional full MaleCNS artifact absent')
    m = json.loads((directory / 'manifest.json').read_text())
    counts = np.load(directory / 'counts.npy', mmap_mode='r')
    ptr = np.load(directory / 'indptr.npy', mmap_mode='r')
    assert counts.dtype == np.int64
    assert counts.sum() == m['retained_synapses']
    assert len(ptr) == m['neurons'] + 1 and ptr[-1] == m['edges']
    assert m['source_connection_rows'] == m['retained_connection_rows'] + m['boundary_connection_rows'] + m['excluded_internal_rows']


def test_physiology_edit_preserves_learning_and_directory_switch_reloads(tmp_path):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full')
    learned = np.linspace(-.4, .4, 8, dtype=np.float32)
    sim.bridge.set_parameters(learned)
    sim.advance(2)
    sim.configure('connectome', physiology=Physiology(gain=12))
    np.testing.assert_array_equal(sim.bridge.parameters, learned)
    assert sim.steps == 0 and sim.full_brain.total_spikes == 0
    torch.testing.assert_close(sim.full_brain.output_gain[sim.bridge.regions['motor']], torch.full((2,), float(np.exp(learned[5]))))
    other = tmp_path / 'other'
    other.mkdir()
    graph_fixture(other)
    previous = sim.full_brain
    sim.configure('connectome', other / 'full')
    assert sim.full_brain is not previous
    assert sim.full_brain.directory == (other / 'full').resolve()
    assert sim.full_brain.config.gain == 12


@pytest.mark.parametrize('mode', ['posture', 'connectome'])
@pytest.mark.parametrize('saved_environment', [None, {'mode': 'spatial', 'odor': 'B', 'intensity': .95, 'source': [4., 2., .01]}])
def test_applied_policy_matches_training_trajectory(tmp_path, mode, saved_environment):
    manifest = graph_fixture(tmp_path)
    physiology = Physiology(gain=20)
    parameters = np.full(8, .2, dtype=np.float32)
    checkpoint = {'format': 'flylab-physical-v1', 'mode': mode, 'task': 'stand',
                  'parameters': torch.from_numpy(parameters), 'physiology': asdict(physiology),
                  'body_parameters': asdict(BodyParameters()), 'graph_sha256': manifest['graph_sha256'],
                  'mapping_profile': 'muscle-routing-v2'}
    if saved_environment is not None:
        checkpoint['environment'] = saved_environment
    environment = TrialEnvironment(**(saved_environment or {}))
    env = FlyEnv(action_mode='muscle' if mode == 'connectome' else 'posture', max_steps=5, task='stand')
    bridge = NeuromuscularBridge(FullBrain(tmp_path / 'full', physiology)) if mode == 'connectome' else None
    rollout(env, parameters, 42, bridge, environment=environment)
    sim = Simulation()
    sim.environment_mode, sim.odor, sim.intensity = 'spatial', 'none', .1
    sim.source[:] = [-8., 9., .01]
    sim.advance(2)
    sim.load_physical(checkpoint, tmp_path / 'full')
    assert not sim.running and sim.steps == 0
    assert (sim.environment_mode, sim.odor, sim.intensity) == (environment.mode, environment.odor, environment.intensity)
    np.testing.assert_array_equal(sim.source, environment.source)
    sim.advance(5)
    np.testing.assert_array_equal(sim.body.data.qpos, env.body.data.qpos)
    np.testing.assert_array_equal(sim.body.data.act, env.body.data.act)
    if bridge:
        torch.testing.assert_close(sim.full_brain.rates, bridge.brain.rates, rtol=0, atol=0)
        assert sim.history[-1]['neural'] == sim.snapshot()['regions'][2]['activity']
