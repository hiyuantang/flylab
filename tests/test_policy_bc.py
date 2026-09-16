from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flylab.policy_bc import action_loss, score_actions
from flylab.policy_evaluation import evaluate_policy


def dataset():
    return dict(observations={'eye': torch.zeros(3, 1)},
                actions=torch.tensor([[[.2, .4], [.8, .9]], [[.6, .2], [.3, .5]], [[.5, .7], [.8, .2]]]),
                mask=torch.tensor([[True, False], [True, True], [True, False]]),
                weights=torch.ones(3))


class FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.commands = torch.nn.Parameter(torch.full((2, 2), .1))

    def forward(self, observations):
        return self.commands.expand(len(observations['eye']), -1, -1)


def test_action_loss_uses_expert_commands_and_masks_future_padding():
    d = dataset()
    model = FixedPolicy()
    expected = torch.tensor([.1, .3, .5, .1, .2, .4, .4, .6]).mean()
    actual = action_loss(model, d, slice(None), 'cpu')
    torch.testing.assert_close(actual, expected)
    d['actions'][~d['mask']] = 1000
    torch.testing.assert_close(action_loss(model, d, slice(None), 'cpu'), expected)
    actual.backward()
    assert torch.isfinite(model.commands.grad).all()


def test_validation_is_independent_of_microbatch_partition():
    d, model = dataset(), FixedPolicy()
    assert score_actions(model, d, 'cpu', 1) == pytest.approx(score_actions(model, d, 'cpu', 3), abs=1e-7)


def test_physical_evaluation_requires_sustained_success_and_keeps_labels_out_of_policy(monkeypatch):
    from flylab import policy_evaluation
    calls = []

    class Simulation:
        def __init__(self, model, settings):
            self.body = SimpleNamespace(data=SimpleNamespace(qpos=np.array([0., 0., 1.])))
            self.sensors = SimpleNamespace(visual_object=None)
            self.steps = 0

        def advance(self):
            self.steps += 1
            calls.append(self.sensors.visual_object.gesture)

    counts = iter([False, True, True, True, True, True, True, True, True, False])
    monkeypatch.setattr(policy_evaluation, 'PolicySimulation', Simulation)
    monkeypatch.setattr(policy_evaluation, 'evaluate_reference', lambda *args: {'pose_reached': next(counts)})
    result = evaluate_policy(object(), {}, ['point', 'point_right'], seeds=[1], horizon=5, stop=Event())
    assert result['successes'] == 0  # A final-frame-only check would incorrectly pass left pointing.
    assert result['episodes'] == 2
    assert len(calls) == 10  # Every physical command step executed.


def test_dataset_splits_whole_episodes_and_labels_actual_commands(tmp_path, monkeypatch):
    from flylab import policy_training
    from flylab.policy_bc import build_dataset
    placements = []

    def examples(settings, reference, cue, placement, history, chunk, stop):
        placements.append((cue, placement))
        return [({'eye': np.full((1, 1, 1), step, np.float32)},
                 np.array([999.]), np.full((chunk, 1), 888.), min(chunk, 3-step))
                for step in range(3)]

    monkeypatch.setattr(policy_training, 'teacher_examples', examples)
    trainer = SimpleNamespace(directory=tmp_path, stop_event=Event(), publish=lambda **kw: None)
    model = SimpleNamespace(sensors=[SimpleNamespace(name='eye')], action_names=['a'], history=1, chunk=2,
                            schema=lambda: {'test': 'command-labels'})
    meta = dict(seed=42, horizon=3, proportions={'point': 1}, demonstrations_per_gesture=2, validation_episodes=1)
    refs = {'point': SimpleNamespace(commands=np.array([[.1], [.2], [.3]], np.float32))}
    splits, key = build_dataset(trainer, {}, model, meta, refs)
    train, val = splits['train'], splits['validation']
    assert len(placements) == 3
    assert len(train['actions']) == 6
    assert len(val['actions']) == 3
    assert set(train['episode_ids']).isdisjoint(val['episode_ids'])
    assert {x.split(':')[1] for x in train['episode_ids']}.isdisjoint({x.split(':')[1] for x in val['episode_ids']})
    torch.testing.assert_close(train['actions'][:3], torch.tensor([[[.1], [.2]], [[.2], [.3]], [[.3], [0.]]]))
    assert train['mask'][:3].tolist() == [[True, True], [True, True], [True, False]]
    cached, cached_key = build_dataset(trainer, {}, model, meta, refs)
    assert cached_key == key and len(placements) == 3
    torch.testing.assert_close(cached['validation']['actions'], val['actions'])
