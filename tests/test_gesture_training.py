from dataclasses import asdict
import json
from threading import Event

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from flylab.body import FlyBody, BodyParameters
from flylab.connectome_adapter import ConnectomeAdapter
from flylab.full_connectome import FullBrain
from flylab.gesture_scene import HandStimulus, GESTURES, ASSETS
from flylab.gesture_training import GestureSession, GestureTrainer, target_poses, pose_error
from flylab.motor_mapping import MAPPING_PROFILE
from flylab.senses import SensorSuite, SensorySettings
from test_full_connectome import graph_fixture


def settings():
    return {'body': asdict(BodyParameters()), 'mapping_profile': MAPPING_PROFILE,
            'gains': [0.] * 8, 'senses': asdict(SensorySettings(vision_enabled=True, vision_model='compound-retina-v1'))}


def test_rendered_mesh_and_ray_mesh_have_identical_vertices_and_faces():
    for gesture in GESTURES:
        render = json.loads((ASSETS / f'{gesture}.json').read_text())
        lines = (ASSETS / f'{gesture}.obj').read_text().splitlines()
        vertices = [float(x) for line in lines if line.startswith('v ') for x in line.split()[1:]]
        faces = [int(x) - 1 for line in lines if line.startswith('f ') for x in line.split()[1:]]
        assert render['vertices'] == vertices
        assert render['indices'] == faces


def test_3d_hand_changes_retinal_pixels_and_respects_occlusion_and_vision_switch():
    body = FlyBody()
    suite = SensorSuite(SensorySettings(vision_enabled=True, vision_model='compound-retina-v1'))
    pixels = []
    for gesture in GESTURES:
        suite.visual_object = HandStimulus(gesture)
        frame = suite.sample(body, [0, 0, 0])
        # Each eye primarily sees its nearer hand; compare the binocular input.
        pixels.append(np.concatenate([eye['channels']['R1-R6'] for eye in frame['vision']['eyes']]))
    assert all(np.max(np.abs(a - b)) > .01 for a, b in zip(pixels, pixels[1:]))
    hand = HandStimulus('palm')
    rays = np.array([[1., 0., .3]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rgb = np.array([[.1, .2, .3]])
    depth = np.array([1.])  # A surface before the distant hand occludes it.
    hand.sample(np.zeros(3), rays, rgb, depth)
    np.testing.assert_array_equal(rgb, [[.1, .2, .3]])
    disabled = SensorSuite(SensorySettings(vision_enabled=False, vision_model='compound-retina-v1'))
    disabled.visual_object = hand
    assert disabled.sample(body, [0, 0, 0])['vision']['mean'] == [0., 0.]


def test_mujoco_hand_rays_match_rendered_triangles_including_finger_gaps():
    # Independent triangle intersections ensure the sensor sees the actual
    # mesh, not a box or convex hull spanning the gaps between fingers.
    for gesture in GESTURES:
        hand = HandStimulus(gesture)
        mesh = json.loads((ASSETS / f'{gesture}.json').read_text())
        vertices = np.array(mesh['vertices']).reshape(-1, 3) + hand.position
        triangles = vertices[np.array(mesh['indices']).reshape(-1, 3)]
        a, edge1, edge2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        rays = np.array([[220., y, z] for y in np.linspace(-55, 55, 8) for z in np.linspace(20, 170, 8)])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        expected = []
        for ray in rays:
            h = np.cross(ray, edge2)
            determinant = np.einsum('ij,ij->i', edge1, h)
            valid = np.abs(determinant) > 1e-9
            inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
            u = inverse * np.einsum('ij,ij->i', -a, h)
            q = np.cross(-a, edge1)
            v = inverse * (q @ ray)
            t = inverse * np.einsum('ij,ij->i', edge2, q)
            hits = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            expected.append(float(t[hits].min()) if hits.any() else np.inf)
        distances = np.full(len(rays), np.inf)
        hand.sample(np.zeros(3), rays, np.zeros((len(rays), 3)), distances)
        np.testing.assert_allclose(distances, expected, rtol=1e-6, atol=1e-4)


def test_target_generation_is_separate_from_actual_body_and_has_distinct_poses():
    body = FlyBody()
    before = body.data.qpos.copy()
    targets = target_poses(body)
    np.testing.assert_array_equal(body.data.qpos, before)
    assert targets['fist']['height'] < targets['palm']['height'] - .15
    assert targets['point']['feet'][0, 2] > targets['palm']['feet'][0, 2] + .3
    assert targets['point_right']['feet'][3, 2] > targets['palm']['feet'][3, 2] + .3
    assert abs(targets['point_right']['feet'][0, 2] - targets['palm']['feet'][0, 2]) < .01
    assert abs(targets['point']['feet'][3, 2] - targets['palm']['feet'][3, 2]) < .01
    assert np.all(targets['point_both']['feet'][[0, 3], 2] > targets['palm']['feet'][[0, 3], 2] + .3)
    assert pose_error(body, targets['palm'])['loss'] == pytest.approx(0)
    assert pose_error(body, targets['fist'])['loss'] > 0
    assert not pose_error(body, targets['point'])['pose_reached']
    assert all(t['ik_residual_mm'] < .05 for t in targets.values())


def test_adapter_keeps_graph_counts_and_signs_and_changes_neural_dynamics(tmp_path):
    graph_fixture(tmp_path)
    brain = FullBrain(tmp_path / 'full')
    adapter = ConnectomeAdapter(brain)
    counts = np.load(tmp_path / 'full/counts.npy').copy()
    ptr, indices = brain.weights.crow_indices().clone(), brain.weights.col_indices().clone()
    baseline = brain.weights.values().clone()
    drive = torch.tensor([15., 0., 0.])
    brain.advance(drive, 20)
    original = brain.voltage.clone()
    brain.reset()
    adapter.apply(np.ones_like(adapter.parameters))
    brain.advance(drive, 20)
    assert not torch.equal(brain.voltage, original)
    torch.testing.assert_close(brain.weights.crow_indices(), ptr)
    torch.testing.assert_close(brain.weights.col_indices(), indices)
    torch.testing.assert_close(brain.weights.values().sign(), baseline.sign())
    np.testing.assert_array_equal(np.load(tmp_path / 'full/counts.npy'), counts)
    assert brain.weights._nnz() == 1 and len(brain.ids) == 3
    adapter.apply(adapter.initial)
    torch.testing.assert_close(brain.weights.values(), baseline, rtol=0, atol=0)
    torch.testing.assert_close(brain.out_weight, adapter.base_out, rtol=0, atol=0)
    with pytest.raises(ValueError):
        adapter.apply(np.full_like(adapter.parameters, np.nan))


def test_gesture_labels_do_not_command_poses_without_visual_path(tmp_path):
    # Fixture has no visual neurons. Different hand labels must not cause a
    # scripted pose or a hidden gesture-to-muscle route.
    graph_fixture(tmp_path)
    session = GestureSession(tmp_path / 'full', settings(), 2, 42)
    results = [session.rollout(session.adapter.initial, cue, 5, 0, Event(), lambda **_: None)
               for cue in GESTURES]
    for result in results[1:]:
        np.testing.assert_array_equal(result['final_joints'], results[0]['final_joints'])


def test_training_checkpoint_failure_honesty_roundtrip_and_cancel(tmp_path):
    graph_fixture(tmp_path)
    trainer = GestureTrainer(tmp_path / 'checkpoints', tmp_path / 'full')
    trainer.start(settings(), iterations=1, horizon=5)
    trainer.thread.join(30)
    status = trainer.snapshot()
    assert not status['running'] and status['error'] is None
    assert len(status['history']) == 2 and len(status['validation']) == len(GESTURES)
    assert not status['success'] and not status['improved']
    assert status['visual_response_difference_rad'] == 0
    saved = trainer._load(status['checkpoint'])
    assert saved['graph_sha256'] == json.loads((tmp_path / 'full/manifest.json').read_text())['graph_sha256']
    trainer.start(settings(), horizon=5, gesture='palm', checkpoint=status['checkpoint'])
    trainer.thread.join(30)
    assert trainer.snapshot()['phase'] == 'inference complete'
    assert len(trainer.snapshot()['history']) == 2  # Inference retains the saved training evidence.
    invalid = dict(saved, implementation_sha256='different-code')
    torch.save(invalid, tmp_path / 'checkpoints/gesture-incompatible.pt')
    with pytest.raises(ValueError, match='implementation_sha256'):
        trainer._load('gesture-incompatible.pt')
    with pytest.raises(ValueError):
        trainer._load('../outside.pt')
    existing = trainer.checkpoints()
    trainer.start(settings(), iterations=10, horizon=25)
    trainer.stop_event.set()
    trainer.thread.join(30)
    assert trainer.snapshot()['cancelled']
    assert trainer.checkpoints() == existing


def test_gesture_api_validation_and_main_experiment_isolation(tmp_path, monkeypatch):
    import flylab.api as api
    graph_fixture(tmp_path)
    from flylab.gesture_batch import BatchGestureTrainer
    runner = BatchGestureTrainer(tmp_path / 'checkpoints', tmp_path / 'full')
    monkeypatch.setattr(api, 'gestures', runner)
    monkeypatch.setattr(api, 'ready', True)
    before = api.sim.body.data.qpos.copy()
    # No lifespan: never load or save the user's live checkpoint in this test.
    client = TestClient(api.app, raise_server_exceptions=True)
    assert client.post('/api/gestures/start', json={'gesture': 'unknown'}).status_code == 422
    assert client.post('/api/gestures/start', json={'rank': 0}).status_code == 422
    assert client.post('/api/gestures/start', json={'horizon': 0}).status_code == 422
    assert client.post('/api/gestures/start', json={'checkpoint': '../bad.pt'}).status_code == 409
    assert client.get('/api/gestures').json()['running'] is False
    assert client.post('/api/gestures/start', json={'iterations': 1, 'batch_size': 1, 'horizon': 5}).status_code == 200
    runner.thread.join(30)
    np.testing.assert_array_equal(api.sim.body.data.qpos, before)
    assert client.post('/api/gestures/stop').status_code == 200


def test_pair_cues_swap_anatomical_hands_under_sagittal_reflection():
    def vertices(cue):
        return np.array(json.loads((ASSETS / f'{cue}.json').read_text())['vertices']).reshape(-1, 3)
    left, right = vertices('point'), vertices('point_right')
    count = len(left) // 2
    mirrored = np.concatenate((left[count:], left[:count])) * [1, -1, 1]
    np.testing.assert_allclose(mirrored, right, atol=1e-12)
    assert left[:count, 1].mean() < 0 and left[count:, 1].mean() > 0
    assert vertices('point_both')[:count, 2].max() > vertices('fist')[:count, 2].max() + 30


def test_api_accepts_both_front_leg_cues_and_rejects_unknown():
    from flylab.api import HandRequest
    from pydantic import ValidationError
    for cue in GESTURES:
        assert HandRequest(gesture=cue).gesture == cue
    with pytest.raises(ValidationError):
        HandRequest(gesture='back_leg')


def test_facing_layout_improves_retinal_gesture_separation():
    from itertools import combinations
    from flylab.gesture_scene import LEGACY_VERSION
    body = FlyBody()
    suite = SensorSuite(SensorySettings(vision_enabled=True, vision_model='compound-retina-v1'))
    def separation(placement):
        pixels = []
        for cue in GESTURES:
            suite.visual_object = HandStimulus(cue, placement=placement)
            frame = suite.sample(body, [0, 0, 0])
            pixels.append(np.concatenate([eye['channels']['R1-R6'] for eye in frame['vision']['eyes']]))
        return min(np.count_nonzero(np.abs(a - b) > .025) for a, b in combinations(pixels, 2))
    previous = separation({'position': [225., 0., 42.], 'quaternion': [1., 0., 0., 0.], 'version': LEGACY_VERSION})
    current = separation({'position': [110., 0., 85.], 'quaternion': [1., 0., 0., 0.]})
    assert current >= previous * 2


def test_randomized_facing_hands_preserve_clearance_and_human_sides():
    from flylab.gesture_scene import random_placement
    vertices = np.array(json.loads((ASSETS / 'palm.json').read_text())['vertices']).reshape(-1, 3)
    for seed in range(50):
        placement = random_placement(np.random.default_rng(seed))
        rotation = np.empty(9)
        import mujoco
        mujoco.mju_quat2Mat(rotation, np.array(placement['quaternion']))
        world = vertices @ rotation.reshape(3, 3).T + placement['position']
        assert world[:, 0].min() >= 15 - 1e-8
        assert world[:, 2].min() >= 5 - 1e-8
        middle = len(world) // 2
        assert world[:middle, 1].mean() < 0 < world[middle:, 1].mean()
