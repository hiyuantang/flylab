import json
from pathlib import Path
import numpy as np
import pytest
from flylab.connectome import ConnectomeProbe


def fixture_graph(tmp_path):
    np.savez(tmp_path/'probe-graph.npz',body_ids=np.array([10,20]),pre=np.array([0]),post=np.array([1]),counts=np.array([7]))
    (tmp_path/'manifest.json').write_text(json.dumps({'neurons':[{'body_id':10,'type':'A','transmitter':'acetylcholine'},{'body_id':20,'type':'B','transmitter':'gaba'}]}))
    return ConnectomeProbe(tmp_path)


def test_no_spontaneous_spikes_and_pulse_is_reproducible(tmp_path):
    graph=fixture_graph(tmp_path)
    assert graph.pulse(10,amplitude=0)['spikes']==0
    result=graph.pulse(10)
    assert result['spikes']>0 and result==graph.pulse(10)
    assert graph.weights.indices().tolist()==[[1],[0]]  # pre -> post, never reversed
    with pytest.raises(ValueError):graph.pulse(999)


def test_real_import_counts_are_preserved():
    root=Path(__file__).resolve().parents[1]/'data'
    if not (root/'manifest.json').exists():pytest.skip('Bulk data not installed')
    manifest=json.loads((root/'manifest.json').read_text())
    arrays=np.load(root/'probe-graph.npz')
    assert int(arrays['counts'].sum())==manifest['synapse_count']
    assert len(arrays['counts'])==manifest['edge_count']
    assert len(set(arrays['body_ids']))==len(manifest['neurons'])
    assert manifest['boundary_connection_rows']>0
