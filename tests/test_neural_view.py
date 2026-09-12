import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from fastapi.testclient import TestClient
from test_full_connectome import graph_fixture
from flylab.full_connectome import FullBrain
from flylab.neural_view import NeuralView


def fixture(tmp_path):
    graph_fixture(tmp_path)
    path = tmp_path / 'full' / 'neurons.feather'
    rows = feather.read_table(path).to_pylist()
    rows[0]['somaLocation'] = [100, 200, 300]
    rows[1]['somaLocation'] = [400, 500, 600]
    rows[2]['somaLocation'] = None
    feather.write_feather(pa.Table.from_pylist(rows), path)
    return FullBrain(tmp_path / 'full')


def test_layout_preserves_ids_coordinates_and_missing_positions(tmp_path):
    brain = fixture(tmp_path)
    view = NeuralView(brain)
    assert view.metadata()['positioned'] == 2
    assert view.metadata()['missing_positions'] == 1
    assert len(view.layout) == len(brain.ids)*21
    np.testing.assert_array_equal(np.frombuffer(view.layout, dtype='<f8', count=3), brain.ids)
    positions = np.frombuffer(view.layout, dtype='<f4', count=9, offset=24).reshape(3, 3)
    np.testing.assert_array_equal(positions[:2], [[100, 200, 300], [400, 500, 600]])
    assert np.isnan(positions[2]).all()
    assert list(view.vnc) == [0, 1, 1]
    assert view.neuron(30)['neuron']['position'] is None


def test_connections_follow_source_direction_counts_and_live_state(tmp_path):
    brain = fixture(tmp_path)
    view = NeuralView(brain)
    assert view.neuron(10, 'incoming')['total'] == 0
    edge = view.neuron(10, 'outgoing')
    assert edge['total'] == 1 and edge['partners'][0]['body_id'] == 20
    assert edge['partners'][0]['synapses'] == 7
    assert view.neuron(20)['partners'][0]['body_id'] == 10
    assert view.neuron(10, 'outgoing', offset=1)['partners'] == []
    brain.rates[0] = 25.5
    brain.voltage[0] = -55
    brain.spike_counts[0] = 9
    updated = view.neuron(10)
    assert updated['neuron']['rate_hz'] == 25.5
    assert updated['neuron']['voltage'] == -55
    assert updated['neuron']['spike_count'] == 9
    # An adjacency cache must never freeze a neuron's live activity.
    brain.rates[0] = 80
    assert view.neuron(10)['neuron']['rate_hz'] == 80


def test_outgoing_search_handles_empty_rows_and_self_connections(tmp_path):
    brain = fixture(tmp_path)
    np.save(tmp_path/'full/indptr.npy', np.array([0, 0, 2, 3]))
    np.save(tmp_path/'full/indices.npy', np.array([1, 0, 0]))
    np.save(tmp_path/'full/counts.npy', np.array([3, 2, 5]))
    view = NeuralView(brain)
    result = view.neuron(10, 'outgoing', limit=1)
    assert result['total'] == 2 and result['outgoing_synapses'] == 7
    assert result['partners'][0]['body_id'] == 30
    assert view.neuron(10, 'outgoing', offset=1)['partners'][0]['body_id'] == 20
    assert view.neuron(20)['partners'][0]['body_id'] == 20
    assert view.neuron(20, 'outgoing')['partners'][0]['body_id'] == 20


def test_api_binary_snapshots_and_validation_are_read_only(tmp_path, monkeypatch):
    import flylab.api as api
    from flylab.simulation import Simulation
    fixture(tmp_path)
    monkeypatch.setattr(api, 'DATA', tmp_path)
    monkeypatch.setattr(api, 'sim', Simulation())
    monkeypatch.setattr(api, 'ready', False)
    with TestClient(api.app) as client:
        before = api.sim.full_brain.voltage.clone()
        meta = client.get('/api/neural-view').json()
        layout = client.get('/api/neural-view/layout')
        activity = client.get('/api/neural-view/activity')
        assert layout.status_code == activity.status_code == 200
        assert layout.headers['X-Graph-SHA256'] == meta['graph_sha256']
        assert len(layout.content) == 63 and len(activity.content) == 12
        np.testing.assert_array_equal(np.frombuffer(activity.content, dtype='<f4'), api.sim.full_brain.rates.numpy().astype('float32'))
        assert client.get('/api/neural-view/neuron/999').status_code == 404
        assert client.get('/api/neural-view/neuron/10?limit=201').status_code == 422
        assert client.get('/api/neural-view/neuron/10?direction=both').status_code == 422
        assert client.get('/api/neural-view/neuron/10?offset=-1').status_code == 422
        torch.testing.assert_close(before, api.sim.full_brain.voltage, rtol=0, atol=0)
        assert api.sim.steps == 0
