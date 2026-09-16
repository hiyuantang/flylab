from pathlib import Path
from threading import Event
import copy
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from flylab.gesture_batch import BatchGestureTrainer
from flylab.gesture_gradients import GradientBrain, Delivery, muscle_activation
from flylab.gesture_versions import stratified_gestures, descendants
from flylab.gesture_scene import random_placement, HandStimulus
from flylab.connectome_adapter import ConnectomeAdapter
from flylab.full_connectome import FullBrain
from flylab.neuromuscular import NeuromuscularBridge
from flylab.body import FlyBody
from test_full_connectome import graph_fixture
from test_gesture_training import settings as legacy_settings

def settings():
    config = legacy_settings()
    config["body"]["elasticity_profile"] = "stance-elastic-v1"
    return config


def engine(tmp_path):
    graph_fixture(tmp_path)
    brain = FullBrain(tmp_path / 'full')
    adapter = ConnectomeAdapter(brain, rank=3)
    result = GradientBrain(brain, adapter, NeuromuscularBridge(brain))
    result.sync_weights()
    return result


def test_surrogate_forward_matches_reference_and_only_adapter_gets_gradients(tmp_path):
    g = engine(tmp_path)
    b = g.brain
    drive = torch.tensor([15., 0., 0.])
    b.advance(drive, 20)
    reference = [x.clone() for x in (b.voltage, b.current, b.spikes, b.rates)]
    b.reset(); g.reset()
    g.advance(drive)
    for expected, value in zip(reference, (b.voltage, b.current, b.spikes, b.rates)):
        torch.testing.assert_close(expected, value, atol=0, rtol=0)
    b.rates.sum().backward()
    assert torch.isfinite(g.parameters.grad).all() and g.parameters.grad.norm() > 0
    assert not b.weights.requires_grad and b.weights.grad is None
    frozen = g.adapter.base.clone()
    optimizer = torch.optim.Adam([g.parameters], lr=.01)
    optimizer.step(); g.detach_state(); g.sync_weights()
    torch.testing.assert_close(g.adapter.base, frozen, atol=0, rtol=0)
    assert not torch.equal(b.weights.values(), frozen)
    with torch.inference_mode():
        b.reset(); b.advance(drive, 20)
        assert not b.rates.requires_grad


def test_sparse_delivery_backward_matches_dense_derivative(tmp_path):
    g = engine(tmp_path)
    g.sync_weights()
    emitted = torch.tensor([0., 2., .4], dtype=torch.float64, requires_grad=True)
    factors = torch.ones(g.adapter.group_count, g.adapter.group_count, dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([.3, .8, -.7], dtype=torch.float64)
    (Delivery.apply(emitted, factors, g) * upstream).sum().backward()
    torch.testing.assert_close(emitted.grad, g.brain.weights.to_dense().T @ upstream)
    expected = torch.zeros_like(factors).flatten()
    for pre in range(3):
        for edge in range(int(g.brain.out_ptr[pre]), int(g.brain.out_ptr[pre + 1])):
            expected[g.out_pairs[edge]] += upstream[g.brain.out_post[edge]] * emitted.detach()[pre] * g.adapter.base_out[edge]
    torch.testing.assert_close(factors.grad.flatten(), expected)


def test_activation_matches_body_and_has_gradient():
    body = FlyBody()
    for magnitude in [.7, .1, .8]:
        action = torch.linspace(0, magnitude, body.model.nu, dtype=torch.float64, requires_grad=True)
        activation = muscle_activation(action, torch.tensor(body.data.act.copy()), body.model)
        body.step_muscles(action.detach().numpy())
        np.testing.assert_allclose(activation.detach(), body.data.act, atol=1e-12)
        activation.sum().backward()
        assert action.grad.norm() > 0 and torch.isfinite(action.grad).all()


def test_online_stratification_and_transform_reproducibility():
    proportions = {'palm': .7, 'fist': .2, 'point': .1}
    rng = np.random.default_rng(9)
    counts = {g: 0 for g in proportions}
    for _ in range(3000):
        batch = stratified_gestures(rng, proportions, 4)
        for cue in counts:
            n = int(np.sum(batch == cue))
            assert abs(n - 4 * proportions[cue]) < 1.00001
            counts[cue] += n
    assert all(abs(counts[g] / 12000 - proportions[g]) < .01 for g in counts)
    assert set(stratified_gestures(rng, {'palm': 0, 'fist': 1, 'point': 0}, 10)) == {'fist'}
    a, b = np.random.default_rng(23), np.random.default_rng(23)
    poses = []
    for _ in range(30):
        pose = random_placement(a)
        assert pose == random_placement(b)
        assert 100 <= pose['position'][0] <= 140
        assert np.isclose(np.linalg.norm(pose['quaternion']), 1)
        hand = HandStimulus('point', placement=pose)
        np.testing.assert_allclose(hand.data.mocap_quat[0], pose['quaternion'])
        poses.append(pose)
    assert poses[0] != poses[1]


def fixture_demonstrations(session, horizon, stop, cues=None):
    # Three-neuron fixtures test optimizer/serialization plumbing, not whether
    # that deliberately tiny graph can control a complete six-legged body.
    from flylab.gesture_batch import demonstrations
    return demonstrations(session, horizon, stop, cues=cues, available_channels=tuple(range(84)))


def run(trainer, **kw):
    trainer.reference_factory = fixture_demonstrations
    trainer.start(settings(), iterations=1, horizon=5, batch_size=1, **kw)
    trainer.thread.join(30)
    result = trainer.snapshot()
    assert not result['running'] and not result['error']
    return result


def test_batches_publish_once_per_iteration_and_versions_branch_and_delete(tmp_path):
    graph_fixture(tmp_path)
    t = BatchGestureTrainer(tmp_path / 'weights', tmp_path / 'full')
    frames = []
    publish = t.publish
    def record(**values):
        if 'frame' in values:
            frames.append(values['preview_iteration'])
        publish(**values)
    t.publish = record
    first = run(t, proportions={'palm': 0, 'fist': 0, 'point': 1}, rank=3)
    assert first['run_id'].startswith('gesture-run-')
    assert first['training_setup']['checkpoint'] is None
    assert first['training_setup']['rank'] == 3
    assert first['training_setup']['proportions']['point'] == 1
    assert frames == [1] and len(first['history']) == 1
    assert first['preview_gesture'] == 'point' and len(first['muscle_comparison']['names']) == 84
    assert first['gradient_norm'] == 0  # This tiny fixture has no motor outputs.
    root = first['checkpoint']
    payload = t._load(root)
    assert 'optimizer' not in payload and payload['parameters'].numel() < 200
    child = run(t, checkpoint=root)['checkpoint']
    sibling = run(t, checkpoint=root)['checkpoint']
    tree = t.snapshot()['versions']
    assert len({v['chart_id'] for v in tree}) == 1
    assert set(descendants(tree, root)) == {root, child, sibling}
    assert t._load(child)['metadata']['parent'] == root
    assert t._load(child)['rank'] == 3
    with pytest.raises(ValueError, match='active model'):
        t.delete_versions(root, [root, child, sibling], child)
    with pytest.raises(ValueError, match='branch changed'):
        t.delete_versions(root, [root])
    t.delete_versions(child, [child])
    assert set(t.checkpoints()) == {root, sibling}
    t.delete_versions(root, [root, sibling])
    assert t.checkpoints() == []
    assert not list((tmp_path / 'weights').glob('*.pt'))


def test_api_batch_validation_and_cancel_leave_live_state_untouched(tmp_path, monkeypatch):
    import flylab.api as api
    graph_fixture(tmp_path)
    from flylab.gesture_training import Cancelled
    def wait_for_cancel(session, horizon, stop, cues=None):
        assert stop.wait(10)
        raise Cancelled()
    t = BatchGestureTrainer(tmp_path / 'weights', tmp_path / 'full', reference_factory=wait_for_cancel)
    monkeypatch.setattr(api, 'gestures', t)
    monkeypatch.setattr(api, 'ready', True)
    before = api.sim.body.data.qpos.copy()
    client = TestClient(api.app)
    for data in [{'rank':0},{'rank':65},{'batch_size':0},{'learning_rate':0},{'iterations':1001}]:
        assert client.post('/api/gestures/start', json=data).status_code == 422
    for data in [{'proportions':{'palm':0,'fist':0,'point':0}}, {'proportions':{'bad':1}}, {'checkpoint':'../bad.pt'}]:
        assert client.post('/api/gestures/start', json=data).status_code == 409
    assert client.post('/api/gestures/start', json={'rank':3,'batch_size':1,'iterations':100}).status_code == 200
    response = client.post('/api/gestures/stop')
    assert response.json()['stopping']
    t.stop_event.set()  # Backend shutdown cancels, whereas UI Stop pauses at a boundary.
    t.thread.join(30)
    assert t.snapshot()['cancelled'] and not t.checkpoints()
    np.testing.assert_array_equal(api.sim.body.data.qpos, before)


def test_clean_lab_has_no_phantom_visual_ball():
    from dataclasses import replace
    from flylab.senses import SensorySettings, SensorSuite
    body = FlyBody()
    original = SensorySettings(vision_enabled=True, stimulus_position=(5., 0., 2.))
    moved = replace(original, stimulus_position=(5., 3., 2.))
    a = SensorSuite(original).sample(body, [0., 0., 0.])
    b = SensorSuite(moved).sample(body, [0., 0., 0.])
    np.testing.assert_array_equal(a['vision']['pixels'], b['vision']['pixels'])


def test_loaded_adapter_and_hand_survive_reset_and_live_save(tmp_path):
    from flylab.gesture_training import GestureSession
    from flylab.live_state import save_live, load_live
    graph_fixture(tmp_path)
    session = GestureSession(tmp_path / 'full', settings(), 3, 42)
    session.adapter.apply(np.ones_like(session.adapter.initial) * .2)
    sim = session.sim
    sim._gesture_adapter = session.adapter
    sim.gesture_adapter = {'rank':3, 'parameters':torch.from_numpy(session.adapter.parameters.copy())}
    sim.gesture_model = 'gesture-example.pt'
    sim.sensors.visual_object = HandStimulus('point', placement=random_placement(np.random.default_rng(9)))
    expected = sim.full_brain.weights.values().clone()
    sim.reset()
    torch.testing.assert_close(sim.full_brain.weights.values(), expected, atol=0, rtol=0)
    sim.advance()
    save_live(sim, tmp_path / 'state.pt')
    restored = load_live(tmp_path / 'state.pt', tmp_path / 'full')
    assert restored.gesture_model == sim.gesture_model
    assert restored.sensors.visual_object.summary() == sim.sensors.visual_object.summary()
    torch.testing.assert_close(restored.full_brain.weights.values(), expected, atol=0, rtol=0)
    sim.advance(); restored.advance()
    torch.testing.assert_close(restored.full_brain.voltage, sim.full_brain.voltage, atol=0, rtol=0)
    np.testing.assert_array_equal(restored.body.data.qpos, sim.body.data.qpos)


def test_pose_loss_includes_front_and_supporting_leg_channels():
    from flylab.gesture_batch import target_muscle_indices
    body = FlyBody()
    for cue in ['palm', 'fist', 'point', 'point_right', 'point_both']:
        selected = target_muscle_indices(body, cue)
        np.testing.assert_array_equal(selected, np.arange(84))
        actual = torch.zeros(84, requires_grad=True)
        (actual[selected] - 1).square().mean().backward()
        np.testing.assert_array_equal(np.flatnonzero(actual.grad.numpy()), selected)


def test_bilateral_batch_mix_and_targets(tmp_path):
    from flylab.gesture_scene import GESTURES
    mix = {g: float(g in {'point_right', 'point_both'}) for g in GESTURES}
    sampled = stratified_gestures(np.random.default_rng(5), mix, 8)
    assert list(sampled).count('point_right') == list(sampled).count('point_both') == 4
    graph_fixture(tmp_path)
    trainer = BatchGestureTrainer(tmp_path / 'weights', tmp_path / 'full')
    status = run(trainer, proportions={'point_right': 1})
    assert status['preview_gesture'] == 'point_right'
    assert len(status['muscle_comparison']['names']) == 84
    assert {name[:2] for name in status['muscle_comparison']['names']} == {'lf', 'lm', 'lh', 'rf', 'rm', 'rh'}


def test_motor_grouping_refines_soma_side_and_inherits_legacy_weights(tmp_path):
    g=engine(tmp_path)
    b=g.brain
    for row,side,target in zip(b.neurons,['L','R','L'],['Ti flexor MN','Ti flexor MN','Ti extensor MN']):
        row.update(superclass='vnc_motor',rootSide=None,somaSide=side,subclass='fl',type=target)
    old=ConnectomeAdapter(b,rank=2)
    new=ConnectomeAdapter(b,rank=2,grouping='motor-target-v2')
    assert old.group_count==1 and new.group_count==3
    assert len({group[2] for group in new.groups})==3
    parameters=torch.tensor(old.initial)
    parameters[0]=.4
    old.apply(parameters.numpy())
    expected=b.weights.values().clone()
    inherited=new.inherit(old.groups,parameters)
    new.apply(inherited.numpy())
    torch.testing.assert_close(b.weights.values(),expected,atol=0,rtol=0)
    assert inherited.shape==(2,3,2)


@pytest.mark.parametrize('resume_run', [True, False])
def test_stop_at_boundary_preserves_run_or_allows_new_start(tmp_path, resume_run, monkeypatch):
    graph_fixture(tmp_path)
    trainer = BatchGestureTrainer(tmp_path / 'weights', tmp_path / 'full', reference_factory=fixture_demonstrations)
    optimizers = []
    adam = torch.optim.Adam
    def track_adam(*args, **kwargs):
        optimizer = adam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer
    monkeypatch.setattr(torch.optim, 'Adam', track_adam)
    paused = Event()
    publish = trainer.publish
    def record(**values):
        publish(**values)
        if values.get('preview_iteration') == 1:
            trainer.request_stop()
        if values.get('resumable'):
            paused.set()
    trainer.publish = record
    args = dict(iterations=2, horizon=5, batch_size=1, proportions={'point':1}, early_stopping=False)
    trainer.start(settings(), **args)
    try:
        assert paused.wait(30)
        state = trainer.snapshot()
        assert state['resumable'] and not state['running'] and not state['stopping']
        assert len(state['history']) == 1
        assert next(iter(optimizers[0].state.values()))['step'].item() == 1
        old_thread = trainer.thread
        trainer.publish = publish
        if resume_run:
            trainer.resume()
            assert trainer.thread is old_thread
        else:
            trainer.start(settings(), **args)
            assert not old_thread.is_alive() and trainer.thread is not old_thread
        trainer.thread.join(30)
        result = trainer.snapshot()
        assert not result['running'] and not result['error'] and not result['resumable']
        assert [row['iteration'] for row in result['history']] == [1,2]
        saved = trainer._load(result['checkpoint'])
        assert len(optimizers) == (1 if resume_run else 2)
        assert next(iter(optimizers[-1].state.values()))['step'].item() == 2
        assert saved['metadata']['iterations_completed'] == 2
        assert saved['metadata']['best_iteration'] == 0
    finally:
        trainer.stop_event.set()
        trainer.resume_event.set()
        trainer.thread.join(10)


def test_configurable_early_stopping_and_best_initial_export(tmp_path):
    graph_fixture(tmp_path)
    trainer = BatchGestureTrainer(tmp_path / 'weights', tmp_path / 'full', reference_factory=fixture_demonstrations)
    trainer.start(settings(), iterations=3, horizon=5, batch_size=1, proportions={'point':1}, early_stopping_patience=1)
    trainer.thread.join(30)
    result = trainer.snapshot()
    assert not result['error'] and result['early_stopped'] and len(result['history']) == 1
    payload = trainer._load(result['checkpoint'])
    assert payload['metadata']['best_iteration'] == 0
    assert payload['metadata']['validation_loss'] == payload['metadata']['validation_initial_loss']
    assert torch.count_nonzero(payload['parameters'][0]) == 0


def test_surrogate_scale_request_validation():
    from pydantic import ValidationError
    from flylab.api import GestureRequest
    for value in [0, -0.1, 1.1, float('nan'), float('inf')]:
        with pytest.raises(ValidationError):
            GestureRequest(surrogate_scale=value)
    assert GestureRequest(surrogate_scale=.01).surrogate_scale == .01


def test_surrogate_scale_is_recorded_and_applied(tmp_path):
    from test_full_connectome import graph_fixture
    graph_fixture(tmp_path)
    trainer = BatchGestureTrainer(tmp_path/'weights', tmp_path/'full', reference_factory=fixture_demonstrations)
    trainer.start(settings(), iterations=1, horizon=5, batch_size=1, surrogate_scale=.01)
    trainer.thread.join(30)
    result = trainer.snapshot()
    assert not result['running'] and not result['error']
    payload = trainer._load(result['checkpoint'])
    assert result['surrogate_scale'] == .01
    assert payload['settings']['surrogate_scale'] == .01
    assert payload['metadata']['surrogate_scale'] == .01
