from dataclasses import asdict
from threading import Event
import json
import numpy as np
import pytest
import torch
from flylab.policy_model import (SensorActionTransformer, SensorSpec, angular_pool,
                                 optical_geometry, observation_window, eye_observation)
from flylab.policy_simulation import PolicySimulation
from flylab.policy_training import PolicyTrainer, example_loss, teacher_examples
from flylab.body import FlyBody, BodyParameters
from flylab.senses import SensorySettings
from flylab.gesture_targets import muscle_reference
from flylab.gesture_scene import random_placement


def settings():
    return {'body': asdict(BodyParameters()), 'senses': asdict(SensorySettings(
        vision_enabled=True, vision_model='compound-retina-balanced-v4', calibration_sphere=False))}


def model():
    body = FlyBody()
    return SensorActionTransformer(action_names=[body.model.actuator(i).name for i in range(84)], width=32, layers=1)


def inputs(batch=2):
    return {name: torch.rand(batch, 8, count, 9) for name, count in [('left_eye', 1024), ('right_eye', 1024)]}


def test_equal_tokens_preserve_every_measured_facet():
    for axes, _ in optical_geometry():
        pool = angular_pool(axes, 64)
        assert pool.shape == (64, len(axes))
        np.testing.assert_allclose(pool.sum(1), 1, atol=1e-6)
        assert (pool.sum(0) > 0).all()
        assert np.all((pool > 0).sum(0) == 1)


def test_schema_roundtrip_and_modality_extension():
    torch.manual_seed(4)
    policy = SensorActionTransformer(sensors=[SensorSpec('touch', 6)], width=32, layers=1)
    data = {'touch': torch.rand(2, 8, 6, 1)}
    restored = SensorActionTransformer.from_schema(policy.schema())
    restored.load_state_dict(policy.state_dict())
    assert torch.equal(policy(data), restored(data))
    with pytest.raises(ValueError, match='sensor names'):
        policy({'gesture': torch.ones(2, 1)})


def test_batch_parallel_matches_independent_predictions_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    policy = model()
    data = inputs()
    batched = policy(data)
    separate = torch.cat([policy({k: v[i:i+1] for k,v in data.items()}) for i in range(2)])
    torch.testing.assert_close(batched, separate, atol=1e-6, rtol=1e-5)
    assert batched.shape == (2, 5, 84)
    assert (batched >= 0).all() and (batched <= 1).all()
    batched.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


def test_real_optics_forward_and_muscle_physics():
    policy = model()
    sim = PolicySimulation(policy, settings())
    assert sim.full_brain is None and sim.bridge is None
    before = sim.body.data.act.copy()
    sim.advance()
    snapshot = sim.snapshot()
    assert snapshot['model']['controller'] == 'transformer'
    assert snapshot['model']['parameters'] > 0 and snapshot['model']['neurons'] == 0
    assert sim.time == pytest.approx(.02)
    assert np.isfinite(sim.body.data.qpos).all()
    assert np.max(np.abs(sim.body.data.act - before)) > 0
    assert np.all(sim.body.data.ctrl[84:] == 0)
    assert len(sim.visual_history) == 1
    sim.reset()
    assert not sim.visual_history


def test_supervised_activation_loss_learns_and_matches_teacher():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    policy = model()
    ref = muscle_reference('palm', 10)
    examples = teacher_examples(settings(), ref, 'palm', random_placement(np.random.default_rng(4)), 8, 5, Event())
    examples = examples[::3]
    initial = float(example_loss(policy, examples, ref.body.model, 'cpu').detach())
    opt = torch.optim.Adam(policy.parameters(), lr=.002)
    for _ in range(15):
        opt.zero_grad()
        loss = example_loss(policy, examples, ref.body.model, 'cpu')
        loss.backward()
        opt.step()
    assert float(example_loss(policy, examples, ref.body.model, 'cpu').detach()) < initial * .6
    class Teacher(torch.nn.Module):
        chunk = 5
        def forward(self, x):
            # First window: replay exactly the teacher commands.
            return torch.tensor(ref.commands[:5, :84], dtype=torch.float32)[None]
    assert float(example_loss(Teacher(), examples[:1], ref.body.model, 'cpu')) < 1e-10


def test_checkpoint_lineage_load_and_branch_deletion(tmp_path):
    from flylab.policy_training import policy_versions, policy_digest
    policy = model()
    trainer = PolicyTrainer(tmp_path)
    def save(name, parent, version):
        metadata = dict(id=name, chart_id='chart-test', chart_name='Test policy', parent=parent,
                        version=version, created_at=f'2026-09-14T00:00:0{version}Z',
                        policy_sha256=policy_digest(), controller_kind='transformer')
        trainer._save(policy, policy.state_dict(), settings(), metadata)
    save('policy-one.pt', None, 1)
    save('policy-two.pt', 'policy-one.pt', 2)
    assert len(policy_versions(tmp_path)) == 2
    restored = trainer.load_simulation('policy-two.pt')
    data = inputs(1)
    torch.testing.assert_close(policy(data), restored.policy(data))
    with pytest.raises(ValueError, match='Invalid policy'):
        trainer._load('../policy-one.pt')
    with pytest.raises(ValueError, match='branch changed'):
        trainer.delete_versions('policy-one.pt', ['policy-one.pt'])
    with pytest.raises(ValueError, match='active model'):
        trainer.delete_versions('policy-one.pt', ['policy-one.pt','policy-two.pt'], 'policy-two.pt')
    save('policy-three.pt', 'policy-one.pt', 3)
    save('policy-four.pt', 'policy-two.pt', 4)
    save('policy-five.pt', 'policy-four.pt', 5)
    branch = ['policy-two.pt', 'policy-four.pt', 'policy-five.pt']
    with pytest.raises(ValueError, match='branch changed'):
        trainer.delete_versions('policy-two.pt', ['policy-two.pt'])
    with pytest.raises(ValueError, match='active model'):
        trainer.delete_versions('policy-two.pt', branch, 'policy-five.pt')
    assert set(trainer.delete_versions('policy-two.pt', branch)['deleted']) == set(branch)
    assert {v['id'] for v in policy_versions(tmp_path)} == {'policy-one.pt', 'policy-three.pt'}
    assert not (tmp_path / 'policy-four.json').exists()
    assert not (tmp_path / 'policy-five.pt').exists()
    assert len(trainer.delete_versions('policy-one.pt', ['policy-one.pt','policy-three.pt'])['deleted']) == 2
    assert not list(tmp_path.glob('*.pt'))


def test_pause_resume_keeps_current_run():
    import threading
    trainer = PolicyTrainer('/private/tmp/unused-policy-test')
    trainer.status['running'] = True
    trainer.request_stop()
    assert trainer.status['stopping']
    trainer.thread = threading.Thread(target=trainer._pause_boundary)
    trainer.thread.start()
    import time
    deadline = time.monotonic() + 2
    while not trainer.status.get('resumable') and time.monotonic() < deadline:
        time.sleep(.01)
    trainer.resume()
    trainer.thread.join(2)
    assert not trainer.thread.is_alive()
    assert trainer.status['running'] and not trainer.status['resumable']


def test_training_failure_is_visible_and_restorable(tmp_path):
    trainer = PolicyTrainer(tmp_path)
    trainer.start({'body': {'invalid_field': 1}, 'senses': asdict(SensorySettings())}, iterations=1)
    trainer.thread.join(5)
    assert not trainer.status['running']
    assert trainer.status['error']
    assert trainer.status['phase'] == 'Training failed'


def test_api_status_with_visual_controller(monkeypatch):
    from flylab import api
    sim = PolicySimulation(model(), settings())
    monkeypatch.setattr(api, 'sim', sim)
    monkeypatch.setattr(api, 'ready', True)
    assert api.current_snapshot()['model']['controller'] == 'transformer'
    assert not api.current_snapshot()['policy_activity']['measured']
    sim.advance()
    activity = api.current_snapshot()['policy_activity']
    assert activity['measured'] and activity['step'] == 1
    assert len(activity['values']['muscles']) == 84
    sim.reset()
    assert not api.current_snapshot()['policy_activity']['measured']
    assert api.execution_status()['current'] is None
    assert api.health()['status'] == 'ok'
    with pytest.raises(ValueError, match='Load another model'):
        sim.configure('connectome')
    with pytest.raises(ValueError, match='compound-eye'):
        sim.configure_senses(SensorySettings(vision_model='legacy-grid-v2'))


def test_new_run_after_pause_uses_selected_parent_and_new_setup(tmp_path, monkeypatch):
    import threading
    import time
    from flylab.gesture_training import Cancelled
    from flylab import policy_training
    trainer = PolicyTrainer(tmp_path)
    old_finished = threading.Event()

    def old_worker():
        try:
            trainer._pause_boundary()
        except Cancelled:
            pass
        finally:
            old_finished.set()

    trainer.status['running'] = True
    trainer.request_stop()
    trainer.thread = threading.Thread(target=old_worker)
    trainer.thread.start()
    deadline = time.monotonic() + 2
    while not trainer.status.get('resumable') and time.monotonic() < deadline:
        time.sleep(.01)
    assert trainer.status['resumable']
    selected = {'schema': model().schema(), 'settings': settings(), 'metadata': {
        'chart_id': 'selected-chart', 'chart_name': 'Selected', 'seed': 7,
        'learning_rate': .002, 'training_recipe': 'action-chunk-bc-v1'}}
    loads, runs = [], []

    def load(name):
        loads.append(name)
        return selected

    def new_worker(run_settings, saved, proportions, metadata):
        assert old_finished.is_set()
        assert not trainer.stop_event.is_set()
        assert not trainer.pause_requested.is_set()
        runs.append((saved, metadata))
        trainer.publish(running=False, phase='Training complete')

    monkeypatch.setattr(trainer, '_load', load)
    monkeypatch.setattr(trainer, '_run_policy', new_worker)
    monkeypatch.setattr(policy_training, 'policy_device', lambda: 'cpu')
    trainer.start(settings(), checkpoint='policy-selected.pt', iterations=17, batch_size=8,
                  horizon=45, seed=7, learning_rate=.002, proportions={'point': 1},
                  early_stopping=False, early_stopping_patience=6, version_name='New branch')
    trainer.thread.join(2)
    assert not trainer.thread.is_alive()
    assert loads == ['policy-selected.pt']
    assert len(runs) == 1 and runs[0][0] is selected
    assert runs[0][1]['parent'] == 'policy-selected.pt'
    assert trainer.status['run_id'] == runs[0][1]['id']
    setup = trainer.snapshot(False)['training_setup']
    assert setup['checkpoint'] == 'policy-selected.pt'
    assert setup['iterations'] == 17 and setup['batch_size'] == 8
    assert setup['horizon'] == 45 and setup['seed'] == 7
    assert setup['early_stopping'] is False and setup['early_stopping_patience'] == 6
    assert setup['version_name'] == 'New branch' and setup['proportions'] == {'point': 1}
    assert not trainer.status.get('resumable')
