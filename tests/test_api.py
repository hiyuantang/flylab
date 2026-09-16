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


def test_timestep_switch_is_saved_without_reset_and_invalid_values_rejected(tmp_path):
    with TestClient(app) as client:
        before = client.post('/api/control', json={'action': 'step'}).json()
        assert client.post('/api/execution', json={'device': 'cpu', 'neural_dt_ms': 2}).status_code == 422
        response = client.post('/api/execution', json={'device': 'cpu', 'neural_dt_ms': 1})
        assert response.status_code == 200, response.text
        execution = response.json()['current']
        assert execution['neural_dt_ms'] == 1 and execution['effective_delay_ms'] == 2
        assert execution['effective_refractory_ms'] == 3
        assert len(list((tmp_path / 'execution-backups').glob('*.pt'))) == 1
        after = client.post('/api/control', json={'action': 'live_restore'}).json()
        assert after['time'] == before['time']
        assert after['model']['spikes'] == before['model']['spikes']
        assert after['timing']['neural_dt_ms'] == 1


def test_timestep_switch_rolls_back_if_save_fails(monkeypatch):
    import flylab.api as api
    original = api.save_live
    def fail_live(sim, path):
        if path.name == 'live-state.pt':
            raise OSError('test storage failure')
        return original(sim, path)
    with TestClient(app) as client:
        with monkeypatch.context() as patch:
            patch.setattr(api, 'save_live', fail_live)
            response = client.post('/api/execution', json={'device': 'cpu', 'neural_dt_ms': 1})
            assert response.status_code == 400
            assert api.sim.full_brain.config.dt_ms == .1
            assert api.sim.full_brain.tick_index == 0
            assert api.sim.full_brain.execution_history == []


def test_60hz_api_switch_preserves_state_and_reports_rate():
    import math
    with TestClient(app) as client:
        before = client.post('/api/control', json={'action': 'step'}).json()
        bad = client.post('/api/execution', json={'device': 'cpu', 'muscle_command_hz': 60, 'neural_dt_ms': .1})
        assert bad.status_code == 400
        assert client.get('/api/execution').json()['current']['muscle_command_hz'] == 50
        response = client.post('/api/execution', json={'device': 'cpu', 'muscle_command_hz': 60})
        assert response.status_code == 200, response.text
        assert math.isclose(1000 / response.json()['current']['neural_dt_ms'], 960)
        after = client.get('/api/state').json()
        assert after['time'] == before['time'] and after['body'] == before['body']
        after = client.post('/api/control', json={'action': 'step'}).json()
        assert after['time'] == pytest.approx(.02 + 1/60)
        assert after['timing']['muscle_command_hz'] == 60
        assert 'performance' in after['timing']


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


@pytest.mark.parametrize('profile,channels', [('muscle-routing-v4', 186), ('muscle-routing-v5', 196)])
@pytest.mark.parametrize('coupling', ['serial', 'pipelined'])
def test_peripheral_upgrade_preserves_gains_and_rolls_back_failed_save(tmp_path, monkeypatch, profile, channels, coupling):
    import flylab.api as api
    import numpy as np
    import torch
    from flylab.live_state import NEURAL_TENSORS
    with TestClient(app) as client:
        client.post('/api/connectome/mapping', json={'profile': 'muscle-routing-v3'}).raise_for_status()
        api.sim.set_coupling(coupling)
        api.sim.bridge.set_parameters(np.linspace(-.1, .1, 8))
        client.post('/api/control', json={'action': 'step'}).raise_for_status()
        old_body, old_bridge = api.sim.body, api.sim.bridge
        clock = api.sim.time
        api.sim.pending_command[:] = np.linspace(0, 1, old_body.model.nu, dtype=np.float32)
        buffered = api.sim.pending_command.copy()
        timestamps = api.sim.coupling_summary()
        neural = {k: getattr(api.sim.full_brain, k).clone() for k in NEURAL_TENSORS}
        save = api.save_live
        def fail_final(sim, path):
            if path.name == 'live-state.pt':
                raise OSError('test save failure')
            return save(sim, path)
        monkeypatch.setattr(api, 'save_live', fail_final)
        response = client.post('/api/connectome/mapping', json={'profile': profile})
        assert response.status_code == 400
        assert api.sim.body is old_body and api.sim.bridge is old_bridge
        np.testing.assert_array_equal(api.sim.pending_command, buffered)
        for key, value in neural.items():
            torch.testing.assert_close(value, getattr(api.sim.full_brain, key), rtol=0, atol=0)
        monkeypatch.setattr(api, 'save_live', save)
        response = client.post('/api/connectome/mapping', json={'profile': profile})
        response.raise_for_status()
        assert api.sim.body.model.nu == channels and api.sim.time == clock
        np.testing.assert_array_equal(api.sim.pending_command[:len(buffered)], buffered)
        assert np.count_nonzero(api.sim.pending_command[len(buffered):]) == 0
        assert api.sim.coupling_summary() == timestamps
        for record in response.json()['mapping'] + response.json()['unmapped']:
            assert record['sources'] and record['confidence']['identity']['level']
            assert record['confidence']['mechanics']['level'] in {'approximate', 'unresolved'}
        np.testing.assert_array_equal(api.sim.bridge.parameters, old_bridge.parameters)
        for key, value in neural.items():
            torch.testing.assert_close(value, getattr(api.sim.full_brain, key), rtol=0, atol=0)
        for name, index in old_body.body_ids.items():
            np.testing.assert_allclose(old_body.data.xpos[index], api.sim.body.data.xpos[api.sim.body.body_ids[name]], atol=1e-12, rtol=0)
        assert client.post('/api/control', json={'action': 'live_restore'}).json()['time'] == clock
        assert api.sim.bridge.mapping_profile == profile
        if coupling == 'pipelined':
            np.testing.assert_array_equal(api.sim.pending_command[:len(buffered)], buffered)
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


def test_fp16_live_api_preserves_clock_restores_and_validates():
    import torch
    import flylab.api as api
    if not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable')
    with TestClient(app) as client:
        before = client.post('/api/control', json={'action': 'step'}).json()
        response = client.post('/api/execution', json={'device': 'mps', 'precision': 'float16', 'muscle_command_hz': 60})
        assert response.status_code == 200, response.text
        current = response.json()['current']
        assert current['precision'] == 'torch.float16'
        assert current['kernel'] == 'metal-active-rows-f16-v1-mark4'
        assert current['muscle_command_hz'] == 60
        assert client.get('/api/state').json()['time'] == before['time']
        assert client.post('/api/execution', json={'device': 'mps'}).json()['current']['precision'] == 'torch.float16'
        client.post('/api/control', json={'action': 'live_restore'})
        assert api.sim.full_brain.dtype == torch.float16
        assert client.post('/api/control', json={'action': 'step'}).status_code == 200
        assert client.post('/api/execution', json={'device': 'cpu', 'precision': 'float16'}).status_code == 400
        assert api.sim.full_brain.dtype == torch.float16
        assert client.post('/api/execution', json={'device': 'mps', 'precision': 'float8'}).status_code == 422
        client.post('/api/control', json={'action': 'reset'})
        assert client.get('/api/health').json()['precision'] == 'torch.float16'


def test_pipeline_switch_backup_restore_and_save_failure_rollback(tmp_path, monkeypatch):
    import flylab.api as api
    import numpy as np
    with TestClient(app) as client:
        before=client.post('/api/control',json={'action':'step'}).json()
        response=client.post('/api/execution',json={'device':'cpu','coupling_mode':'pipelined'})
        assert response.status_code==200,response.text
        assert response.json()['current']['coupling_mode']=='pipelined'
        assert client.get('/api/state').json()['time']==before['time']
        assert client.post('/api/control',json={'action':'live_restore'}).status_code==200
        assert api.sim.coupling_mode=='pipelined'
        client.post('/api/control',json={'action':'step'})
        pending=api.sim.pending_command.copy()
        save=api.save_live
        def fail_live(sim,path):
            if path==tmp_path/'live-state.pt':raise OSError('save failed')
            return save(sim,path)
        with monkeypatch.context() as patch:
            patch.setattr(api,'save_live',fail_live)
            response=client.post('/api/execution',json={'device':'cpu','coupling_mode':'serial'})
            assert response.status_code==400
        assert api.sim.coupling_mode=='pipelined'
        np.testing.assert_array_equal(pending,api.sim.pending_command)
        assert client.post('/api/execution',json={'device':'cpu','coupling_mode':'serial'}).status_code==200
        assert client.post('/api/execution',json={'device':'cpu','coupling_mode':'anything'}).status_code==422


def test_reward_api_signed_scores_stale_episode_and_stop(tmp_path,monkeypatch):
    import flylab.api as api
    from test_dopamine import reward_graph
    # Use a separate graph because the autouse fixture already built tmp_path/full.
    directory=tmp_path/'reward';directory.mkdir();reward_graph(directory)
    monkeypatch.setattr(api,'DATA',directory)
    with TestClient(app) as client:
        state=client.get('/api/state').json()
        assert state['reward']['available']
        episode=state['episode']
        assert client.post('/api/control',json={'action':'reward','score':2}).status_code==422
        assert client.post('/api/control',json={'action':'reward','score':1,'episode':episode+1}).status_code==400
        response=client.post('/api/control',json={'action':'reward','score':-.5,'episode':episode})
        assert response.status_code==200
        assert response.json()['reward']['last_event']['score']==-.5
        assert response.json()['reward']['pulse_remaining_s']==.2
        assert client.post('/api/control',json={'action':'reward','score':1}).status_code==400
        stopped=client.post('/api/control',json={'action':'reward_stop','episode':episode}).json()
        assert not stopped['reward']['enabled'] and stopped['reward']['pulse_remaining_s']==0
        assert client.post('/api/control',json={'action':'reward','score':1}).status_code==200
        reset=client.post('/api/control',json={'action':'reset'}).json()
        assert reset['reward']['events']==0 and not reset['reward']['enabled']
        assert client.post('/api/control',json={'action':'reward','score':1,'episode':episode}).status_code==400


def test_all_neuron_evidence_read_only_and_paginated():
    import flylab.api as api
    with TestClient(app) as client:
        voltage = api.sim.full_brain.voltage.clone()
        before = api.sim.time
        metadata = client.get('/api/evidence')
        assert metadata.status_code == 200
        assert metadata.json()['total'] == 3
        records = []
        for offset in range(3):
            response = client.get(f'/api/evidence/neurons?offset={offset}&limit=1')
            assert response.status_code == 200
            records.extend(response.json()['records'])
        assert len({r['body_id'] for r in records}) == 3
        for row in records:
            response = client.get(f"/api/evidence/neurons/{row['body_id']}")
            assert response.status_code == 200
            assert response.json()['annotations']['bodyId'] == row['body_id']
            assert len(response.json()['claims']) == 5
        assert client.get('/api/evidence/neurons?limit=101').status_code == 422
        assert client.get('/api/evidence/neurons?offset=-1').status_code == 422
        assert client.get('/api/evidence/neurons?scope=invalid').status_code == 422
        assert client.get('/api/evidence/neurons/999999999').status_code == 404
        import torch
        assert torch.equal(voltage, api.sim.full_brain.voltage)
        assert api.sim.time == before
