from fastapi.testclient import TestClient
from flylab.api import app
import pytest
from test_full_connectome import graph_fixture


@pytest.fixture(autouse=True)
def isolated_measured_workbench(tmp_path, monkeypatch):
    import flylab.api as api
    from flylab.simulation import Simulation
    graph_fixture(tmp_path)
    monkeypatch.setattr(api, 'DATA', tmp_path)
    monkeypatch.setattr(api, 'sim', Simulation())
    monkeypatch.setattr(api, 'ready', False)


def test_controls_validation_and_live_stream():
    with TestClient(app) as client:
        assert client.get('/api/health').json()['engine']=='PyTorch + MuJoCo'
        client.post('/api/control',json={'action':'reset'})
        before=client.get('/api/state').json()
        response=client.post('/api/control',json={'action':'step'})
        assert response.json()['steps']==before['steps']+1
        assert client.post('/api/control',json={'action':'speed','speed':100}).status_code==422
        assert client.post('/api/control',json={'action':'stimulate','region':'unknown'}).status_code==422
        with client.websocket_connect('/ws') as ws:
            message=ws.receive_json()
            assert message['simulation']['model']['measured_connectome'] is True
            assert message['simulation']['model']['precision'] == 'torch.float64'


def test_research_meshes_are_served():
    with TestClient(app) as client:
        response=client.get('/models/c_head.stl')
        assert response.status_code==200
        assert len(response.content)>80000


def test_environment_switch_resets_and_reports_spatial_sensation():
    with TestClient(app) as client:
        client.post('/api/control', json={'action': 'step'})
        state = client.post('/api/control', json={
            'action': 'environment', 'environment': 'spatial',
        }).json()
        assert state['steps'] == 0 and not state['running']
        assert state['environment']['mode'] == 'spatial'
        assert len(state['environment']['antennae']) == 2
        assert 0 < state['environment']['concentration'] < state['intensity']
        assert client.post('/api/control', json={
            'action': 'environment', 'environment': 'unknown',
        }).status_code == 422
        state = client.post('/api/control', json={
            'action': 'environment', 'environment': 'uniform',
        }).json()
        assert state['environment']['concentration'] == state['intensity']


def test_physical_controller_and_mechanics_validation():
    with TestClient(app) as client:
        assert client.post('/api/control', json={'action': 'controller', 'controller': 'posture'}).status_code == 422
        assert client.post('/api/control', json={'action': 'controller', 'controller': 'synthetic'}).status_code == 422
        state = client.post('/api/control', json={'action': 'controller', 'controller': 'connectome'}).json()
        assert state['model']['controller'] == 'connectome'
        assert state['model']['neurons'] == 3
        state = client.post('/api/control', json={'action': 'step'}).json()
        assert state['steps'] == 1 and state['body']['contacts'] > 0
        mechanics = client.get('/api/mechanics').json()
        assert len(mechanics['joints']) == 70
        assert client.post('/api/mechanics', json={'mass_scale': 0}).status_code == 422
        assert client.post('/api/physical/start', json={'mode': 'unknown'}).status_code == 422
        assert client.post('/api/control', json={'action': 'physical_load', 'checkpoint': '../outside.pt'}).status_code == 400


def test_missing_graph_never_runs_synthetic_fallback(tmp_path, monkeypatch):
    import flylab.api as api
    monkeypatch.setattr(api, 'DATA', tmp_path / 'absent')
    with TestClient(app) as client:
        state = client.get('/api/state').json()
        assert not state['model']['ready'] and state['model']['neurons'] == 0
        assert client.post('/api/control', json={'action': 'run'}).status_code == 400


def test_live_save_resume_restores_full_brain_and_body_time():
    with TestClient(app) as client:
        client.post('/api/control', json={'action': 'step'})
        saved = client.post('/api/control', json={'action': 'live_save'}).json()
        client.post('/api/control', json={'action': 'step'})
        restored = client.post('/api/control', json={'action': 'live_restore'}).json()
        assert restored['time'] == saved['time'] and restored['steps'] == saved['steps']
        assert restored['model']['spikes'] == saved['model']['spikes']
        assert not restored['running']


def test_sensory_controls_reset_validate_and_stream_current_pose():
    with TestClient(app) as client:
        before = client.post('/api/control', json={'action': 'step'}).json()
        state = client.post('/api/control', json={'action': 'sensing', 'senses': {
            'vision_enabled': True, 'hearing_enabled': True, 'wind_enabled': True,
        }}).json()
        assert state['steps'] == 0 and state['episode'] > before['episode']
        assert state['senses']['vision']['mean'][0] > 0
        assert min(state['senses']['hearing']) > 0 and min(state['senses']['wind']) > 0
        assert client.post('/api/control', json={'action': 'sensing', 'senses': {'sound_frequency': 0}}).status_code == 422
        assert client.post('/api/control', json={'action': 'sensing', 'senses': {'stimulus_position': [0, 0, -5]}}).status_code == 400
        assert client.get('/api/state').json()['episode'] == state['episode']
        dark = client.post('/api/control', json={'action': 'sensing', 'senses': {'illumination': 0}}).json()
        assert dark['senses']['vision']['mean'] == [0, 0]
        client.post('/api/control', json={'action': 'sensing', 'senses': {}})


def test_anatomical_explorer_reads_full_state_and_validates_queries():
    with TestClient(app) as client:
        before = client.get('/api/state').json()
        root = client.get('/api/anatomy').json()
        assert root['group']['count'] == 3
        assert sum(g['count'] for g in root['items']) == 3
        group = next(g['id'] for g in root['items'] if g['label'] == 'vnc_motor')
        neurons = client.get('/api/anatomy', params={'group': group, 'view': 'neurons'}).json()
        assert [r['body_id'] for r in neurons['items']] == [20, 30]
        wiring = client.get('/api/anatomy', params={'group': group, 'view': 'wiring'}).json()['wiring']
        assert wiring['incoming'] == {'connections': 1, 'synapses': 7}
        assert client.get('/api/anatomy', params={'group': 'invalid'}).status_code == 404
        assert client.get('/api/anatomy', params={'limit': 101}).status_code == 422
        assert client.get('/api/anatomy', params={'offset': -1}).status_code == 422
        assert client.get('/api/anatomy', params={'view': 'invented'}).status_code == 422
        after = client.get('/api/state').json()
        assert (after['steps'], after['model']['spikes']) == (before['steps'], before['model']['spikes'])


def test_execution_validates_and_preserves_state_when_gpu_unavailable(monkeypatch):
    import torch
    with TestClient(app) as client:
        current = client.get('/api/execution').json()
        assert current['current']['device'] == 'cpu'
        before = client.post('/api/control', json={'action': 'step'}).json()
        assert client.post('/api/execution', json={'device': 'cuda'}).status_code == 422
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
        assert client.post('/api/execution', json={'device': 'mps'}).status_code == 400
        after = client.get('/api/state').json()
        assert after['time'] == before['time']
        assert after['model']['device'] == 'cpu'
        assert after['model']['spikes'] == before['model']['spikes']


def test_gpu_execution_switch_and_restore_keep_time_and_status(tmp_path):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable')
    with TestClient(app) as client:
        before = client.post('/api/control', json={'action': 'step'}).json()
        switched = client.post('/api/execution', json={'device': 'mps'})
        assert switched.status_code == 200, switched.text
        assert switched.json()['current']['precision'] == 'torch.float32'
        assert client.get('/api/health').json()['device'] == 'mps'
        assert len(list((tmp_path / 'execution-backups').glob('*.pt'))) == 1
        restored = client.post('/api/control', json={'action': 'live_restore'}).json()
        assert restored['model']['device'] == 'mps' and restored['time'] == before['time']
        after = client.post('/api/control', json={'action': 'step'}).json()
        assert after['steps'] == before['steps'] + 1
        assert client.get('/api/anatomy').status_code == 200
        assert client.post('/api/execution', json={'device': 'cpu'}).status_code == 200
        assert client.get('/api/state').json()['time'] == after['time']


def test_mapping_upgrade_preserves_full_state_and_backs_up(tmp_path):
    import flylab.api as api
    import numpy as np
    import torch
    from flylab.live_state import NEURAL_TENSORS
    with TestClient(app) as client:
        client.post('/api/connectome/mapping', json={'profile': 'leg-routing-v1'}).raise_for_status()
        client.post('/api/control', json={'action': 'step'}).raise_for_status()
        before = {k: getattr(api.sim.full_brain, k).clone() for k in NEURAL_TENSORS}
        pose, clock = api.sim.body.data.qpos.copy(), api.sim.time
        response = client.post('/api/connectome/mapping', json={'profile': 'muscle-routing-v2'})
        assert response.status_code == 200
        assert response.json()['mapping_profile'] == 'muscle-routing-v2'
        assert not api.sim.running and api.sim.time == clock
        np.testing.assert_array_equal(pose, api.sim.body.data.qpos)
        for key, value in before.items():
            torch.testing.assert_close(value, getattr(api.sim.full_brain, key), rtol=0, atol=0)
        assert len(list((tmp_path / 'mapping-backups').glob('live-*.pt'))) == 2
        restored = client.post('/api/control', json={'action': 'live_restore'})
        assert restored.status_code == 200 and restored.json()['time'] == clock
        assert api.sim.bridge.mapping_profile == 'muscle-routing-v2'
        assert client.post('/api/connectome/mapping', json={'profile': 'unknown'}).status_code == 422


def test_peripheral_upgrade_preserves_gains_and_rolls_back_failed_save(tmp_path, monkeypatch):
    import flylab.api as api
    import numpy as np
    import torch
    from flylab.live_state import NEURAL_TENSORS
    with TestClient(app) as client:
        client.post('/api/connectome/mapping', json={'profile': 'muscle-routing-v3'}).raise_for_status()
        api.sim.bridge.set_parameters(np.linspace(-.1, .1, 8))
        client.post('/api/control', json={'action': 'step'}).raise_for_status()
        old_body, old_bridge = api.sim.body, api.sim.bridge
        clock = api.sim.time
        neural = {k: getattr(api.sim.full_brain, k).clone() for k in NEURAL_TENSORS}
        save = api.save_live
        def fail_final(sim, path):
            if path.name == 'live-state.pt':
                raise OSError('test save failure')
            return save(sim, path)
        monkeypatch.setattr(api, 'save_live', fail_final)
        response = client.post('/api/connectome/mapping', json={'profile': 'muscle-routing-v4'})
        assert response.status_code == 400
        assert api.sim.body is old_body and api.sim.bridge is old_bridge
        for key, value in neural.items():
            torch.testing.assert_close(value, getattr(api.sim.full_brain, key), rtol=0, atol=0)
        monkeypatch.setattr(api, 'save_live', save)
        response = client.post('/api/connectome/mapping', json={'profile': 'muscle-routing-v4'})
        response.raise_for_status()
        assert api.sim.body.model.nu == 186 and api.sim.time == clock
        np.testing.assert_array_equal(api.sim.bridge.parameters, old_bridge.parameters)
        for key, value in neural.items():
            torch.testing.assert_close(value, getattr(api.sim.full_brain, key), rtol=0, atol=0)
        for name, index in old_body.body_ids.items():
            np.testing.assert_allclose(old_body.data.xpos[index], api.sim.body.data.xpos[api.sim.body.body_ids[name]], atol=1e-12, rtol=0)
        assert client.post('/api/control', json={'action': 'live_restore'}).json()['time'] == clock
        assert api.sim.bridge.mapping_profile == 'muscle-routing-v4'
        wall = next(j for j in client.get('/api/mechanics').json()['joints'] if 'pharyngeal' in j['name'])
        assert wall['range_unit'] == 'mm' and wall['range'] == [0., .025]


def test_compound_eye_upgrade_preserves_neural_body_and_checkpoint(tmp_path, monkeypatch):
    import flylab.api as api
    import torch
    import numpy as np
    from flylab.live_state import NEURAL_TENSORS
    with TestClient(app) as client:
        client.post('/api/control', json={'action': 'step'}).raise_for_status()
        before={k:getattr(api.sim.full_brain,k).clone() for k in NEURAL_TENSORS}
        body=api.sim.body; qpos=body.data.qpos.copy(); clock=api.sim.time
        old_sensors=api.sim.sensors
        save=api.save_live
        def fail_final(sim, path):
            if path.name == 'live-state.pt':
                raise OSError('test retinal save failure')
            return save(sim,path)
        monkeypatch.setattr(api,'save_live',fail_final)
        failed=client.post('/api/control',json={'action':'vision_upgrade'})
        assert failed.status_code==400 and api.sim.sensors is old_sensors
        monkeypatch.setattr(api,'save_live',save)
        response=client.post('/api/control',json={'action':'vision_upgrade'})
        response.raise_for_status()
        assert api.sim.sensors.settings.vision_model=='compound-retina-v1'
        assert api.sim.body is body and api.sim.time==clock and not api.sim.running
        np.testing.assert_array_equal(qpos,body.data.qpos)
        for k,v in before.items(): torch.testing.assert_close(v,getattr(api.sim.full_brain,k),rtol=0,atol=0)
        assert len(list((tmp_path/'sensory-backups').glob('*.pt')))==2
        restored=client.post('/api/control',json={'action':'live_restore'})
        restored.raise_for_status()
        assert restored.json()['senses']['vision']['model']=='compound-retina-v1'
        assert api.sim.time==clock
