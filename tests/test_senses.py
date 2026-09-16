from dataclasses import asdict, replace
import json

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch

from flylab.body import FlyBody, BodyParameters
from flylab.env import FlyEnv
from flylab.full_connectome import build_full, FullBrain, Physiology
from flylab.neuromuscular import NeuromuscularBridge
from flylab.physical_policy import TrialEnvironment
from flylab.physical_training import rollout
from flylab.senses import SensorSuite, SensorySettings
from flylab.simulation import Simulation
from flylab.scenes import get_scene


def test_eyes_follow_pose_and_light_without_mutating_body():
    body = FlyBody()
    suite = SensorSuite(SensorySettings(vision_enabled=True, calibration_sphere=True))
    before = body.data.qpos.copy()
    frame = suite.sample(body, [12., 0., .01])
    assert np.asarray(frame['vision']['pixels']).shape == (2, 8, 16)
    assert frame['vision']['mean'][0] != frame['vision']['mean'][1]
    assert suite.sample(body, [12., 0., .01]) == frame
    np.testing.assert_array_equal(body.data.qpos, before)
    json.dumps(frame, allow_nan=False)
    dark = SensorSuite(replace(suite.settings, illumination=0)).sample(body, [12., 0., .01])
    assert not np.any(dark['vision']['pixels'])
    body.data.qpos[3:7] = [0., 0., 0., 1.]
    mujoco.mj_forward(body.model, body.data)
    rotated = suite.sample(body, [12., 0., .01])
    assert not np.array_equal(frame['vision']['pixels'], rotated['vision']['pixels'])


def test_sound_frequency_distance_and_modality_ablation():
    body = FlyBody()
    settings = SensorySettings(hearing_enabled=True, wind_enabled=True)
    frame = SensorSuite(settings).sample(body, [0., 0., .01])
    assert min(frame['hearing']) > 0  # A 250 Hz carrier must not alias to silence at 50 Hz.
    assert min(frame['wind']) > 0
    far = SensorSuite(replace(settings, stimulus_position=(60., 30., 1.5))).sample(body, [0., 0., .01])
    off_frequency = SensorSuite(replace(settings, sound_frequency=1000.)).sample(body, [0., 0., .01])
    assert max(far['hearing']) < min(frame['hearing'])
    assert max(off_frequency['hearing']) < min(frame['hearing'])
    off = SensorSuite(SensorySettings(touch_enabled=False, proprioception_enabled=False)).sample(body, [0., 0., .01], intensity=0)
    for name in ('hearing', 'wind', 'touch', 'proprioception', 'odor'):
        assert not np.any(off[name])
    for values in ({'sound_frequency': 0}, {'stimulus_position': (0., 0., -1.)}, {'illumination': float('nan')}):
        with pytest.raises(ValueError):
            SensorySettings(**values)


def sensory_graph(path):
    rows = [
        {'bodyId': 1, 'superclass': 'ol_sensory', 'class': 'visual', 'type': 'R1-R6', 'rootSide': 'L', 'subclass': None},
        {'bodyId': 2, 'superclass': 'cb_sensory', 'class': 'mechanosensory', 'type': 'JO-A1', 'rootSide': 'L', 'subclass': 'auditory'},
        {'bodyId': 3, 'superclass': 'cb_sensory', 'class': 'mechanosensory', 'type': 'JO-EV1', 'rootSide': 'R', 'subclass': 'wind_gravity'},
        {'bodyId': 4, 'superclass': 'cb_sensory', 'class': 'mechanosensory', 'type': 'JO-unclear', 'rootSide': 'L', 'subclass': None},
        {'bodyId': 5, 'superclass': 'ol_intrinsic', 'class': 'visual', 'type': 'L1', 'rootSide': 'L', 'subclass': None},
    ]
    feather.write_feather(pa.Table.from_pylist(rows), path / 'body-annotations.feather')
    feather.write_feather(pa.table({'body_pre': [1, 2], 'body_post': [5, 5], 'weight': [4, 2]}), path / 'connectome-weights.feather')
    feather.write_feather(pa.table({'body': [1, 2, 3, 4, 5], 'consensus_nt': ['histamine'] + ['acetylcholine'] * 4,
                                  'predicted_nt_confidence': [.9] * 5}), path / 'body-neurotransmitters.feather')
    return build_full(path)


def test_senses_enter_only_annotated_channels_and_histamine_transmits(tmp_path):
    sensory_graph(tmp_path)
    brain = FullBrain(tmp_path / 'full')
    bridge = NeuromuscularBridge(brain)
    body = FlyBody()
    frame = SensorSuite(SensorySettings(vision_enabled=True, hearing_enabled=True, wind_enabled=True)).sample(body, [12., 0., .01])
    drive = bridge.sensory_drive(body, [12., 0., .01], frame=frame)
    assert torch.all(drive[:3] > 0)
    assert torch.count_nonzero(drive[3:]) == 0  # Unclear JO and interneurons get no invented sensory input.
    visual_only = torch.zeros(5)
    visual_only[0] = 21  # 3 threshold gaps, expressed in the paper profile's mV.
    brain.advance(visual_only, 40)
    assert brain.rates[0] > 0 and brain.current[4] < 0
    assert brain.rates[4] == 0  # Histamine is not incorrectly made excitatory.
    counts = bridge.summary()['sensory_neurons']
    assert (counts['visual'], counts['auditory'], counts['wind_gravity']) == (1, 1, 1)


@pytest.mark.parametrize("scene_id", ["lab", "kitchen"])
def test_sensory_checkpoint_reproduces_body_and_neural_trajectory(tmp_path, scene_id):
    manifest = sensory_graph(tmp_path)
    settings = SensorySettings(vision_enabled=True, hearing_enabled=True, wind_enabled=True, sound_frequency=350)
    scene = get_scene(scene_id)
    settings = replace(settings, stimulus_position=tuple(scene['sound_source']))
    environment = TrialEnvironment(senses=settings, scene_id=scene_id, mode='spatial', source=tuple(scene['odor_source']))
    env = FlyEnv(action_mode='muscle', task='stand', max_steps=5)
    bridge = NeuromuscularBridge(FullBrain(tmp_path / 'full'))
    parameters = np.zeros(8, dtype=np.float32)
    rollout(env, parameters, 42, bridge, environment=environment)
    sim = Simulation()
    sim.load_physical({'format': 'flylab-physical-v1', 'mode': 'connectome', 'task': 'stand',
                       'parameters': torch.from_numpy(parameters), 'physiology': asdict(Physiology.paper()),
                       'body_parameters': asdict(BodyParameters()), 'environment': environment.to_dict(),
                       'graph_sha256': manifest['graph_sha256']}, tmp_path / 'full')
    sim.advance(5)
    np.testing.assert_array_equal(sim.body.data.qpos, env.body.data.qpos)
    torch.testing.assert_close(sim.full_brain.rates, bridge.brain.rates, rtol=0, atol=0)
    assert sim.sensors.settings.sound_frequency == 350
    observation = env._observation()
    assert observation.shape == (598,) and env.observation_space.contains(observation)
    assert np.any(observation[-260:-4]) and np.any(observation[-4:])
