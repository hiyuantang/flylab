"""Continuation must follow the same optimization trajectory as uninterrupted work."""
import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from flylab.body import BodyParameters, FlyBody
from flylab.senses import SensorySettings
from flylab.policy_model import SensorActionTransformer
from flylab.policy_training import PolicyTrainer
from flylab.policy_checkpoint import capture_training, restore_training, continuation_snapshot


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_adam_rng_roundtrip_and_lr_override(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(41)

    def update(model, optimizer, rng):
        x = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    update(model, optimizer, rng)
    snapshot = capture_training(model, optimizer, rng, global_step=1,
        validation_loss=.2, stale=0, early_stopping_reference_loss=.2)
    path = tmp_path / 'state.pt'
    torch.save(snapshot, path)
    loaded = torch.load(path, weights_only=True)
    original_snapshot = copy.deepcopy(snapshot)
    expected_loss = update(model, optimizer, rng)
    restored = torch.nn.Linear(3, 2)
    opt2 = torch.optim.Adam(restored.parameters(), lr=.09)
    rng2 = np.random.default_rng(99)
    restore_training(loaded, restored, opt2, rng2, learning_rate=.001)
    assert update(restored, opt2, rng2) == expected_loss
    assert_tree_equal(restored.state_dict(), model.state_dict())
    assert_tree_equal(opt2.state_dict(), optimizer.state_dict())
    assert_tree_equal(snapshot, original_snapshot)  # No live tensor aliases.

    restore_training(loaded, restored, opt2, rng2, learning_rate=.0001)
    assert opt2.param_groups[0]['lr'] == .0001
    assert_tree_equal(opt2.state_dict()['state'], loaded['optimizer_state_dict']['state'])
    assert all(state['step'].item() == 1 for state in opt2.state.values())


@pytest.fixture
def training_setup(tmp_path, monkeypatch, request):
    from flylab import policy_training
    device = getattr(request, 'param', 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable in this process')
    monkeypatch.setattr(policy_training, 'policy_device', lambda: device)
    torch.set_num_threads(1)
    torch.manual_seed(9)
    parameters = BodyParameters(appendage_model='peripheral-v2', elasticity_profile='stance-elastic-v1')
    settings = dict(body=asdict(parameters), senses=asdict(SensorySettings(
        vision_enabled=True, vision_model='compound-retina-v1', calibration_sphere=False)))
    body = FlyBody(parameters)
    model = SensorActionTransformer(action_names=[body.model.actuator(i).name for i in range(84)],
                                    width=16, layers=1)
    trainer = PolicyTrainer(tmp_path)
    metadata = dict(id='policy-seed.pt', chart_id='test', chart_name='Test', parent=None,
        version=1, created_at='2026-09-15T00:00:00Z', learning_rate=.001, seed=42)
    trainer._save(model, model.state_dict(), settings, metadata)
    return trainer, settings


def run(trainer, settings, checkpoint, iterations, **overrides):
    options = dict(checkpoint=checkpoint, iterations=iterations, horizon=25, batch_size=32,
                   learning_rate=.001, proportions={'palm': 1}, early_stopping=False,
                   demonstrations_per_gesture=1, validation_episodes=1, rollout_episodes=1, evaluation_interval=1)
    options.update(overrides)
    trainer.start(settings, **options)
    trainer.thread.join(120)
    assert not trainer.thread.is_alive()
    assert trainer.status.get('error') is None, trainer.status
    return trainer._load(trainer.status['checkpoint'])


@pytest.mark.parametrize('training_setup', ['cpu', 'mps'], indirect=True)
def test_real_trainer_split_matches_uninterrupted(training_setup):
    trainer, settings = training_setup
    whole = run(trainer, settings, 'policy-seed.pt', 4)
    first = run(trainer, settings, 'policy-seed.pt', 2)
    second = run(trainer, settings, first['metadata']['id'], 2)
    for point in ('latest', 'best'):
        assert_tree_equal(whole['training_state'][point], second['training_state'][point])
    assert second['results']['validation_initial_loss'] == first['results']['validation_loss']
    assert second['results']['validation_continued']
    assert second['metadata']['global_step'] == 4
    assert second['metadata']['parent_global_step'] == 2
    assert second['metadata']['optimizer_start'].startswith('Restored Adam')
    assert 'legacy checkpoint' in first['metadata']['optimizer_start']


def test_best_latest_and_changed_lr_use_matching_state(training_setup):
    trainer, settings = training_setup
    first = run(trainer, settings, 'policy-seed.pt', 1)
    later = run(trainer, settings, first['metadata']['id'], 2)
    # Build an iteration-boundary fixture whose latest differs from its best.
    later['training_state']['best'] = first['training_state']['best']
    later['state_dict'] = later['training_state']['best']['state_dict']
    torch.save(later, trainer.directory / later['metadata']['id'])
    for point, step in [('best', 1), ('latest', 3)]:
        child = run(trainer, settings, later['metadata']['id'], 1,
                    resume_from=point, learning_rate=.0001)
        assert child['metadata']['parent_global_step'] == step
        assert child['metadata']['global_step'] == step + 1
        assert child['results']['validation_initial_loss'] == later['training_state'][point]['validation_loss']
        opt = child['training_state']['latest']['optimizer_state_dict']
        assert opt['param_groups'][0]['lr'] == .0001
        assert all(state['step'].item() == step + 1 for state in opt['state'].values())
    with pytest.raises(ValueError, match='Resume point'):
        trainer.start(settings, checkpoint=later['metadata']['id'], resume_from='missing')


def test_changed_seed_restarts_sampling_and_validation(training_setup):
    trainer, settings = training_setup
    first = run(trainer, settings, 'policy-seed.pt', 1)
    child = run(trainer, settings, first['metadata']['id'], 1, seed=77)
    assert not child['results']['validation_continued']
    assert child['metadata']['rng_start'] == 'Restarted sample stream with requested seed'
    assert child['metadata']['optimizer_start'].startswith('Restored Adam')
    assert child['metadata']['validation_signature'] != first['metadata']['validation_signature']


def test_early_stopping_progress_survives_resume(training_setup):
    trainer, settings = training_setup
    first = run(trainer, settings, 'policy-seed.pt', 1)
    first['training_state']['latest']['stale'] = 3
    torch.save(first, trainer.directory / first['metadata']['id'])
    child = run(trainer, settings, first['metadata']['id'], 5,
                early_stopping=True, early_stopping_patience=3)
    assert child['metadata']['completed_iterations'] == 0
    assert child['metadata']['global_step'] == 1
    assert child['results']['history'] == []
    assert_tree_equal(child['training_state']['latest'], first['training_state']['latest'])


def test_old_and_future_checkpoint_formats():
    assert continuation_snapshot({'state_dict': {}}, 'latest') is None
    with pytest.raises(ValueError, match='Unsupported'):
        continuation_snapshot({'training_state': {'version': 999}}, 'latest')


def test_api_routes_resume_point_only_to_visual_trainer(monkeypatch):
    from threading import RLock
    from fastapi.testclient import TestClient
    from flylab import api
    recorded = []

    class Backend:
        lock = RLock()

        def start(self, settings, **kwargs):
            recorded.append(kwargs)

        def snapshot(self, include_frame=True):
            return {'running': False}

    monkeypatch.setattr(api, 'policies', Backend())
    monkeypatch.setattr(api, 'gestures', Backend())
    monkeypatch.setattr(api, 'ready', True)
    monkeypatch.setattr(api.sim, 'running', False)
    monkeypatch.setattr(api, 'combined_training_status', lambda *_: {'running': True})
    client = TestClient(api.app)  # No live-workbench lifespan.
    url = '/api/gestures/start?controller_kind=transformer'
    assert client.post(url, json={'resume_from': 'best', 'learning_rate': .0001}).status_code == 200
    assert recorded[-1]['resume_from'] == 'best'
    assert recorded[-1]['learning_rate'] == .0001
    assert client.post(url, json={'resume_from': 'unknown'}).status_code == 422
    assert client.post('/api/gestures/start', json={}).status_code == 200
    assert 'resume_from' not in recorded[-1]


def test_changed_objective_keeps_weights_but_resets_adam(training_setup):
    trainer, settings = training_setup
    parent = run(trainer, settings, 'policy-seed.pt', 2)
    parent['metadata']['training_recipe'] = 'legacy-activation-mse'
    torch.save(parent, trainer.directory / parent['metadata']['id'])
    child = run(trainer, settings, parent['metadata']['id'], 1)
    assert 'changed to action-chunk' in child['metadata']['optimizer_start']
    assert not child['results']['validation_continued']
    assert child['results']['validation_initial_loss'] == parent['training_state']['latest']['validation_loss']
    assert child['metadata']['global_step'] == 3
    assert all(s['step'].item() == 1 for s in child['training_state']['latest']['optimizer_state_dict']['state'].values())
