"""The target renderer must match supervised channels without mutating the live fly."""
from dataclasses import asdict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flylab import api
from flylab.body import BodyParameters, FlyBody
from flylab.gesture_training import implementation_digest
from flylab.gesture_targets import muscle_reference, TARGET_VERSION


@pytest.mark.parametrize('steps', [10, 25])
@pytest.mark.parametrize('cue', ['palm', 'fist', 'point', 'point_right', 'point_both'])
def test_target_poses_are_reached_with_replayable_50hz_commands(cue, steps):
    parameters = BodyParameters(appendage_model='peripheral-v2', elasticity_profile='stance-elastic-v1')
    reference = muscle_reference(cue, steps, parameters)
    assert reference.evidence['pose_reached'], reference.evidence
    available = reference.evidence['commanded_channels']
    assert len(available) == 52
    missing = [i for i in range(84) if i not in available]
    np.testing.assert_array_equal(reference.commands[:, missing], 0)
    np.testing.assert_array_equal(reference.activations[:, missing], 0)
    replay = FlyBody(parameters)
    for command, activation in zip(reference.commands, reference.activations):
        replay.step_muscles(command, dt=.02)
        np.testing.assert_allclose(replay.data.act, activation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.data.qpos, reference.body.data.qpos, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(reference.commands[:, 84:], 0)
    result = api.target_response(cue, steps, parameters)
    assert result['validation']['target_version'] == TARGET_VERSION
    assert result['validation']['pose_reached']
    assert result['selected_muscles'] == list(range(84))
    np.testing.assert_allclose(result['body']['muscle_activation'], reference.activations[-1], atol=1e-12)


def test_invalid_reference_is_rejected_before_training():
    from types import SimpleNamespace
    from threading import Event
    from flylab.gesture_batch import demonstrations
    session = SimpleNamespace(settings={'body': asdict(BodyParameters())})
    with pytest.raises(ValueError, match='reference|cannot construct'):
        demonstrations(session, 10, Event())


def test_extended_target_and_api_are_isolated_and_validate_requests():
    before = api.sim.body.data.qpos.copy(), api.sim.body.data.act.copy(), api.sim.time
    digest = implementation_digest()
    parameters = BodyParameters(appendage_model='peripheral-v2')
    client = TestClient(api.app)  # Do not start or replace the global neural workbench.
    payload = {'gesture': 'point_right', 'steps': 2, 'parameters': asdict(parameters)}
    response = client.post('/api/gestures/target', json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result['body']['muscle_activation']) == 196
    assert result['selected_muscles'] == list(range(84))
    assert not result['validation']['pose_reached']  # Too short to validate a hold.
    assert client.post('/api/gestures/target', json={**payload, 'gesture': 'unknown'}).status_code == 422
    assert client.post('/api/gestures/target', json={**payload, 'steps': 251}).status_code == 422
    assert client.post('/api/gestures/target', json={**payload, 'steps': 0}).status_code == 422
    assert client.post('/api/gestures/target', json={**payload, 'parameters': {'mass_scale': 0}}).status_code == 422
    np.testing.assert_array_equal(api.sim.body.data.qpos, before[0])
    np.testing.assert_array_equal(api.sim.body.data.act, before[1])
    assert api.sim.time == before[2]
    assert implementation_digest() == digest


def test_training_preview_retains_parent_mechanics(monkeypatch):
    import threading
    recorded = asdict(BodyParameters(strength_scale=1.7, appendage_model='peripheral-v2'))

    class PreviewTrainer:
        lock = threading.RLock()

        def _load(self, name):
            assert name == 'gesture-parent.pt'
            return {'settings': {'body': recorded}}

        def start(self, settings, **options):
            assert options['checkpoint'] == 'gesture-parent.pt'

        def snapshot(self, include_frame=True):
            return {'frame': {'steps': 5}, 'versions': [], 'running': False}

    trainer = PreviewTrainer()
    monkeypatch.setattr(api, 'gestures', trainer)
    monkeypatch.setattr(api, 'ready', True)
    client = TestClient(api.app)
    response = client.post('/api/gestures/start', json={'checkpoint': 'gesture-parent.pt'})
    assert response.status_code == 200
    assert client.get('/api/gestures').json()['frame']['body_parameters'] == recorded
    # Changing the live body's settings must not relabel this training reference.
    monkeypatch.setattr(api.sim, 'body', FlyBody(BodyParameters()))
    assert client.get('/api/gestures').json()['frame']['body_parameters'] == recorded


def test_reference_route_asset_matches_measured_motor_annotations():
    from pathlib import Path
    import json
    import pyarrow.feather as feather
    from flylab.body import LEGS, ACTIVE_DOF
    from flylab.motor_mapping import resolve_motor
    graph=Path(__file__).resolve().parents[1]/'data/full'
    if not (graph/'neurons.feather').exists():
        pytest.skip('Bulk MaleCNS annotations not installed')
    asset=json.loads((Path(api.__file__).parent/'assets/gesture-target-routes.json').read_text())
    assert asset['graph_sha256']==json.loads((graph/'manifest.json').read_text())['graph_sha256']
    rows=[row for row in feather.read_table(graph/'neurons.feather').to_pylist()
          if row.get('superclass') in {'vnc_motor','cb_motor'}]
    for profile,expected in asset['profiles'].items():
        actual=set()
        for row in rows:
            record=resolve_motor(row,profile)
            if record['status']=='mapped':
                for link,axis,sign,coefficient in record['projections']:
                    if link!='pretarsus':
                        actual.add((LEGS.index(record['leg'])*7+ACTIVE_DOF.index((link,axis)))*2+sign)
        assert sorted(actual)==expected
