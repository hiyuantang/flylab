from fastapi.testclient import TestClient
from flylab.api import app


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
            assert message['simulation']['model']['measured_connectome'] is False


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
