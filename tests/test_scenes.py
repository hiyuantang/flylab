"""Cross-check world geometry, physical support, optics and experiment replay."""
import copy
import json

import mujoco
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from flylab.body import FlyBody, BodyParameters
from flylab.env import FlyEnv
from flylab.physical_policy import TrialEnvironment
from flylab.physical_training import PhysicalTrainer, rollout
from flylab.scenes import SCENE_IDS, get_scene
from flylab.senses import SensorSuite, SensorySettings
from flylab.simulation import Simulation


@pytest.mark.parametrize('scene_id', SCENE_IDS)
def test_shared_solids_and_supported_spawn(scene_id):
    scene = get_scene(scene_id)
    json.dumps(scene, allow_nan=False)
    body = FlyBody(scene_id=scene_id)
    baseline = FlyBody()
    assert body.model.nbody == baseline.model.nbody == 70
    assert body.model.ngeom == baseline.model.ngeom + len(scene['objects'])
    assert body.model.body_mass.sum() == pytest.approx(baseline.model.body_mass.sum())
    ids = set()
    for obj in scene['objects']:
        assert obj['id'] not in ids
        ids.add(obj['id'])
        g = body.model.geom(obj['id'])
        assert body.model.geom_bodyid[g.id] == 0
        np.testing.assert_allclose(g.pos, obj['position'])
        np.testing.assert_allclose(g.size[:len(obj['size'])], obj['size'])
        np.testing.assert_allclose(g.quat, obj['quaternion'], atol=1e-12)
        assert np.linalg.norm(obj['quaternion']) == pytest.approx(1)
        assert all(v > 0 for v in obj['size'])
    neutral = body.data.qpos[body.qadr].copy()
    for _ in range(15):
        body.step_posture(neutral, np.zeros(8))
    assert scene['spawn'][2] + .7 < body.data.qpos[2] < scene['spawn'][2] + 1.3
    assert 8 < body.foot_feedback().sum() < 13  # Weight supported in µN, no kinematic pin.
    if scene_id != 'lab':
        assert not any(c.geom1 == 0 or c.geom2 == 0 for c in body.data.contact)
        assert any(body.model.geom_bodyid[c.geom1] == 0 or body.model.geom_bodyid[c.geom2] == 0 for c in body.data.contact)
    assert not any(body.data.warning.number)


def test_retinal_occlusion_is_the_physical_counter():
    body = FlyBody(scene_id='kitchen')
    origin = np.array([*body.scene['spawn'][:2], 1000.])
    ray_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(body.model, body.data, origin, np.array([0., 0., -1.]),
                            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, ray_id)
    assert distance == pytest.approx(100.)
    assert body.model.geom_bodyid[ray_id[0]] == 0
    assert ray_id[0] != 0
    suite = SensorSuite(SensorySettings(vision_enabled=True))
    before = suite.sample(body, body.scene['odor_source'])
    assert before['scene_id'] == 'kitchen'
    assert max(before['vision']['nearest_surface_mm']) < 2
    # Changing a collidable surface's reflectance also changes retinal samples.
    original = body.model.geom_rgba[ray_id[0]].copy()
    try:
        body.model.geom_rgba[ray_id[0], :3] = 0
        after = suite.sample(body, body.scene['odor_source'])
        assert np.mean(after['vision']['pixels']) < np.mean(before['vision']['pixels'])
    finally:
        body.model.geom_rgba[ray_id[0]] = original
    body.data.qpos[3:7] = [0., 0., 0., 1.]
    mujoco.mj_forward(body.model, body.data)
    rotated = suite.sample(body, body.scene['odor_source'])
    assert not np.array_equal(before['vision']['pixels'], rotated['vision']['pixels'])


def test_scene_switch_is_atomic_and_preserves_policy():
    sim = Simulation()
    sim.configure('posture')
    sim.posture_gains[:] = .1
    episode = sim.episode
    sim.configure_scene('bedroom')
    assert sim.episode != episode and not sim.running
    assert sim.body.scene_id == 'bedroom' and sim.environment_mode == 'spatial'
    np.testing.assert_array_equal(sim.posture_gains, np.full(8, .1, dtype=np.float32))
    assert list(sim.source) == get_scene('bedroom')['odor_source']
    assert list(sim.sensors.settings.stimulus_position) == get_scene('bedroom')['sound_source']
    previous = sim.body
    with pytest.raises(ValueError):
        sim.configure_scene('unknown')
    assert sim.body is previous
    sim.configure('posture', body_parameters=BodyParameters(mass_scale=1.1))
    assert sim.body.scene_id == 'bedroom'


def test_room_training_and_checkpoint_replay(tmp_path):
    torch.set_num_threads(1)
    scene = get_scene('kitchen')
    context = TrialEnvironment(scene_id='kitchen', mode='spatial', odor='B', intensity=.3,
                               source=tuple(scene['odor_source']),
                               senses=SensorySettings(vision_enabled=True, hearing_enabled=True,
                                                      stimulus_position=tuple(scene['sound_source'])))
    trainer = PhysicalTrainer(tmp_path, tmp_path / 'no-graph')
    trainer.start(generations=1, population=4, horizon=5, environment=context)
    trainer.thread.join(30)
    status = trainer.snapshot()
    assert not status['running'] and status['error'] is None
    payload = trainer.load(status['checkpoint'])
    assert payload['environment'] == context.to_dict()
    env = FlyEnv(action_mode='posture', task='stand', max_steps=5)
    rollout(env, payload['parameters'].numpy(), 42, environment=context)
    assert env.body.scene_id == 'kitchen' and env.target[2] == 900.5
    sim = Simulation()
    sim.load_physical(payload, tmp_path)
    sim.advance(5)
    np.testing.assert_array_equal(env.body.data.qpos, sim.body.data.qpos)
    assert sim.odor == 'B' and sim.intensity == .3
    invalid = copy.deepcopy(payload)
    invalid['environment']['scene_version'] = 'obsolete-geometry'
    previous = sim.body
    with pytest.raises(ValueError, match='geometry version'):
        sim.load_physical(invalid, tmp_path)
    assert sim.body is previous


def test_scene_api_contract_and_live_training_context(monkeypatch, tmp_path):
    from flylab import api
    from test_full_connectome import graph_fixture
    from flylab.full_connectome import Physiology
    graph_fixture(tmp_path)
    simulation = Simulation()
    simulation.configure('connectome', tmp_path / 'full', Physiology.paper())
    monkeypatch.setattr(api, 'sim', simulation)
    monkeypatch.setattr(api, 'ready', True)
    captured = {}
    monkeypatch.setattr(api.physical, 'start', lambda **kw: captured.update(kw))
    client = TestClient(api.app)
    try:
        assert [s['id'] for s in client.get('/api/scenes').json()] == list(SCENE_IDS)
        definition = client.get('/api/scenes/garden').json()
        assert len(definition['objects']) > 300
        assert client.get('/api/scenes/unknown').status_code == 404
        assert client.post('/api/control', json={'action': 'scene', 'scene_id': 'garden'}).status_code == 200
        state = client.get('/api/state').json()
        assert state['scene']['id'] == state['senses']['scene_id'] == 'garden'
        assert 'objects' not in state['scene']
        assert client.post('/api/control', json={'action': 'scene', 'scene_id': 'missing'}).status_code == 400
        assert client.post('/api/physical/start', json={'mode': 'connectome'}).status_code == 200
        assert captured['environment'].scene_id == 'garden'
        assert captured['environment'].source == tuple(definition['odor_source'])
    finally:
        client.post('/api/control', json={'action': 'scene', 'scene_id': 'lab'})


def test_fly_can_fall_off_the_counter():
    body = FlyBody(scene_id='kitchen')
    body.data.qpos[1] = 600.  # Clear space beyond the counter's front edge (y=850).
    mujoco.mj_forward(body.model, body.data)
    initial_height = float(body.data.qpos[2])
    for _ in range(5):
        body.step_muscles(np.zeros(84))
    assert initial_height - body.data.qpos[2] > 30
    assert body.foot_feedback().sum() == 0
