import numpy as np
import pytest
import torch
import copy
from flylab.body import FlyBody, BodyParameters
from flylab.env import FlyEnv
from flylab.physical_training import PhysicalTrainer, rollout
from flylab.simulation import Simulation
from flylab.senses import SensorySettings, SENSORY_VERSION


def test_reference_envelope_parameters_and_independent_models():
    base = FlyBody()
    fitted = FlyBody(BodyParameters(mass_scale=1.1, strength_scale=.8, limit_mode='reference_envelope'))
    assert fitted.model.body_mass.sum() > base.model.body_mass.sum()
    assert fitted.mechanics()['max_force_parameter_uN'] == 1120
    assert not np.array_equal(fitted.model.jnt_range, base.model.jnt_range)
    assert base.mechanics()['parameters']['mass_scale'] == 1
    assert len(fitted.mechanics()['joints']) == 70
    with pytest.raises(ValueError):
        BodyParameters(strength_scale=float('nan'))


def test_stance_is_physical_and_policy_checkpoint_roundtrip(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    env = FlyEnv(action_mode='posture', task='stand', max_steps=25)
    result = rollout(env, np.zeros(8, dtype=np.float32), seed=5)
    assert result['seconds'] == pytest.approx(.5)
    assert result['minimum_upright'] > .9
    assert env.body.foot_feedback().sum() > 0
    trainer = PhysicalTrainer(tmp_path, tmp_path / 'absent-graph')
    original_save = torch.save
    def checked_save(payload, path):
        assert trainer.checkpoints() == []
        original_save(payload, path)
        assert trainer.checkpoints() == []  # Even complete temporary files are hidden.
    monkeypatch.setattr(torch, 'save', checked_save)
    trainer.start(generations=1, population=4, horizon=5, sensory_settings=SensorySettings(vision_enabled=True, hearing_enabled=True, sound_frequency=350))
    trainer.thread.join(timeout=30)
    status = trainer.snapshot()
    assert not status['running'] and status['error'] is None
    assert status['after']['reward'] >= status['before']['reward'] - 1e-6
    assert len(status['validation']) == 3
    saved = trainer.load(status['checkpoint'])
    assert saved['parameters'].shape == (8,)
    assert saved['format'] == 'flylab-physical-v1'
    assert saved['graph_sha256'] is None
    assert saved['environment']['source'] == [12., 0., .01]
    assert saved['environment']['senses']['sound_frequency'] == 350
    assert saved['environment']['senses']['hearing_enabled']
    assert saved['sensory_version'] == SENSORY_VERSION
    sim = Simulation()
    sim.advance(2)
    sim.running = True
    body, position = sim.body, sim.body.data.qpos.copy()
    for field, invalid in [('parameters', torch.full((8,), float('nan'))), ('mode', 'synthetic'),
                           ('environment', {'intensity': float('nan')}), ('body_parameters', {}),
                           ('physiology', {'dt_ms': -1})]:
        broken = copy.deepcopy(saved)
        broken[field] = invalid
        if field == 'body_parameters':
            del broken[field]
        with pytest.raises(ValueError):
            sim.load_physical(broken, tmp_path)
        assert sim.body is body and sim.controller == 'synthetic' and sim.running and sim.steps == 2
        np.testing.assert_array_equal(sim.body.data.qpos, position)
    with pytest.raises(ValueError):
        trainer.load('../outside.pt')


def test_cancellation_does_not_publish_a_partial_policy(tmp_path):
    trainer = PhysicalTrainer(tmp_path, tmp_path / 'no-graph')
    trainer.start(generations=10, population=4, horizon=100)
    with pytest.raises(ValueError, match='already running'):
        trainer.start()
    trainer.stop_event.set()
    trainer.thread.join(timeout=10)
    result = trainer.snapshot()
    assert not result['running'] and result['cancelled']
    assert result['checkpoint'] is None and trainer.checkpoints() == []


def test_cancellation_during_serialization_does_not_publish(tmp_path, monkeypatch):
    trainer = PhysicalTrainer(tmp_path, tmp_path / 'no-graph')
    original_save = torch.save
    def cancel_at_save(payload, path):
        original_save(payload, path)
        trainer.stop_event.set()
    monkeypatch.setattr(torch, 'save', cancel_at_save)
    trainer.start(generations=1, population=4, horizon=5)
    trainer.thread.join(timeout=30)
    result = trainer.snapshot()
    assert not result['running'] and result['cancelled'] and result['error'] is None
    assert result['checkpoint'] is None and list(tmp_path.iterdir()) == []
