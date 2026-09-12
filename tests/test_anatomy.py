import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch

from flylab.anatomy import AnatomyIndex, group_id
from flylab.full_connectome import FullBrain, build_full


def brain_fixture(path):
    rows = [
        {'bodyId': 10, 'superclass': 'A', 'type': 'alpha', 'somaSide': 'L', 'assignedOlHex1': 1., 'assignedOlHex2': 2.},
        {'bodyId': 20, 'superclass': 'A', 'type': 'alpha', 'somaSide': 'R', 'assignedOlHex1': 1., 'assignedOlHex2': 2.},
        {'bodyId': 30, 'superclass': 'B', 'type': 'beta', 'somaSide': 'L', 'assignedOlHex1': 3., 'assignedOlHex2': 4.},
        {'bodyId': 40, 'superclass': 'B', 'type': None, 'somaSide': None, 'assignedOlHex1': None, 'assignedOlHex2': None},
        {'bodyId': 50, 'superclass': 'B', 'type': 'beta', 'somaSide': 'L', 'assignedOlHex1': 3., 'assignedOlHex2': 4.},
    ]
    feather.write_feather(pa.Table.from_pylist(rows), path / 'body-annotations.feather')
    feather.write_feather(pa.table({'body_pre': [10, 20, 10, 30, 40, 50], 'body_post': [20, 10, 30, 10, 50, 50], 'weight': [3, 5, 7, 11, 13, 17]}), path / 'connectome-weights.feather')
    feather.write_feather(pa.table({'body': [10, 20, 30, 40, 50], 'consensus_nt': ['ach'] * 5, 'predicted_nt_confidence': [.9] * 5}), path / 'body-neurotransmitters.feather')
    build_full(path)
    return FullBrain(path / 'full')


def test_partitions_preserve_identity_unknowns_and_duplicate_column_cells(tmp_path):
    brain = brain_fixture(tmp_path)
    index = brain.anatomy
    def visit(group):
        children = index.partition(group.id)
        if children:
            combined = np.concatenate([g.indices for g in children])
            np.testing.assert_array_equal(np.sort(combined), group.indices)
            assert len(np.unique(combined)) == len(group.indices)
            for child in children:
                assert not child.indices.flags.writeable
                visit(child)
    visit(index.resolve('all'))
    beta_column = next(g for g in index.groups.values() if g.kind == 'column' and g.label == '3, 4')
    np.testing.assert_array_equal(brain.ids[beta_column.indices], [30, 50])
    unknown = [g for g in index.groups.values() if g.kind == 'type' and g.label == 'Unassigned']
    assert len(unknown) == 1 and unknown[0].indices.tolist() == [3]


def test_cross_group_wiring_keeps_direction_weights_and_self_edges(tmp_path):
    brain = brain_fixture(tmp_path)
    groups = brain.anatomy.partition()
    a = next(g for g in groups if g.label == 'A')
    wiring = brain.anatomy.wiring(a.id)
    assert wiring['internal'] == {'connections': 2, 'synapses': 8}
    assert wiring['incoming'] == {'connections': 1, 'synapses': 11}
    assert wiring['outgoing'] == {'connections': 1, 'synapses': 7}
    whole = brain.anatomy.wiring('all')
    assert whole['internal'] == {'connections': 6, 'synapses': 56}
    assert whole['incoming']['connections'] == whole['outgoing']['connections'] == 0
    assert whole['routes'] == []


def test_activity_and_neuron_search_are_real_state_views(tmp_path):
    brain = brain_fixture(tmp_path)
    brain.rates[:] = torch.tensor([0., 2., 4., 6., 8.])
    a = next(g for g in brain.anatomy.partition() if g.label == 'A')
    result = brain.anatomy.snapshot(a.id, 'neurons', '20')
    assert result['group']['mean_hz'] == 1
    assert result['group']['active_neurons'] == 1
    assert result['total'] == 1 and result['items'][0]['body_id'] == 20
    assert result['items'][0]['rate_hz'] == 2
    assert result['items'][0]['voltage_unit'] == 'mV'
    assert len(brain.anatomy.snapshot('all', 'neurons', offset=3, limit=1)['items']) == 1
    assert brain.anatomy.snapshot('all', 'neurons', 'absent')['total'] == 0
    # Direct links resolve in a fresh index without opening ancestors first.
    new_index = AnatomyIndex(brain)
    neuron = new_index.resolve(result['items'][0]['id'])
    assert neuron.indices.tolist() == [1]
    with pytest.raises(ValueError):
        new_index.resolve(group_id(a.path + (('neuron', 30),)))
    for invalid in ['?', 'abc', group_id((('nonsense', 'A'),))]:
        with pytest.raises(ValueError):
            new_index.resolve(invalid)


def test_group_inspection_does_not_change_dynamics(tmp_path):
    brain = brain_fixture(tmp_path)
    control = FullBrain(tmp_path / 'full')
    drive = torch.tensor([20., 0., 0., 0., 0.], dtype=torch.float64)
    brain.advance(drive, 20)
    control.advance(drive, 20)
    for group in brain.anatomy.partition():
        brain.anatomy.snapshot(group.id, 'groups')
        brain.anatomy.snapshot(group.id, 'neurons')
        brain.anatomy.snapshot(group.id, 'wiring')
    brain.advance(drive, 20)
    control.advance(drive, 20)
    for name in ['voltage', 'current', 'rates', 'spike_counts', 'delay_queue', 'last_spike_tick']:
        torch.testing.assert_close(getattr(brain, name), getattr(control, name), rtol=0, atol=0)
    assert brain.time_ms == control.time_ms
    torch.testing.assert_close(brain.weights.values(), control.weights.values(), rtol=0, atol=0)
