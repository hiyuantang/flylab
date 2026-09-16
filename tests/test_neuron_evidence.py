from types import SimpleNamespace

import pytest
import torch

from flylab.full_connectome import Physiology
from flylab.neuron_evidence import NeuronEvidence
from flylab.senses import SensorySettings


def evidence_fixture():
    rows = [
        {'bodyId': 11, 'type': 'IN-test', 'superclass': 'vnc_intrinsic', 'transmitter': 'gaba', 'transmitter_confidence': 0.0},
        {'bodyId': 12, 'type': None, 'superclass': 'cb_sensory', 'class': 'olfactory', 'transmitter': None, 'transmitter_confidence': float('nan')},
        {'bodyId': 13, 'type': 'R7', 'superclass': 'ol_sensory', 'class': 'visual', 'transmitter': 'histamine'},
        {'bodyId': 14, 'type': 'MN-test', 'superclass': 'vnc_motor', 'transmitter': 'acetylcholine'},
    ]
    brain = SimpleNamespace(neurons=rows, lookup={r['bodyId']: i for i, r in enumerate(rows)},
                            config=Physiology.paper(), manifest={'sources': []})
    bridge = SimpleNamespace(motor_inventory=[{'body_id': 14, 'status': 'unmapped'}],
        olfactory=[torch.tensor([1])], visual=[torch.tensor([2])], auditory=[], wind_gravity=[], touch=[], position=[],
        leg_feedback=SimpleNamespace(groups={}), retina=lambda: SimpleNamespace(records=[{'body_id': 13, 'reason': 'spectral_channel_unavailable'}]))
    return NeuronEvidence(brain, bridge)


def test_paging_covers_every_neuron_once_and_filters_unknowns():
    index = evidence_fixture()
    ids = [row['body_id'] for offset in range(0, 4, 2) for row in index.page(offset=offset, limit=2)['records']]
    assert ids == [11, 12, 13, 14]
    assert index.page(scope='sensory')['total'] == 2
    assert index.page(scope='motor')['records'][0]['body_id'] == 14
    assert index.page(query='GABA')['total'] == 1
    assert index.page(superclass='vnc_intrinsic')['total'] == 1
    assert index.page(query='absent')['records'] == []
    assert index.page(offset=99)['records'] == []


def test_claims_do_not_promote_scores_or_missing_fields_to_certainty():
    index = evidence_fixture()
    first = index.detail(11, SensorySettings())
    assert first['transmitter_score'] == 0
    assert first['claims'][1]['level'] == 'dataset-supported anatomy'
    assert first['claims'][3]['level'] == 'model assumption'
    missing = index.detail(12, SensorySettings())
    assert missing['transmitter_score'] is None
    assert missing['claims'][2]['level'] == 'unresolved'
    assert missing['annotations']['type'] is None
    assert index.metadata()['missing_transmitter_score'] == 3
    with pytest.raises(ValueError):
        index.detail(99, SensorySettings())


def test_sensor_profiles_and_unresolved_retina_remain_explicit():
    index = evidence_fixture()
    settings = SensorySettings(vision_model='compound-retina-v1', vision_enabled=True)
    detail = index.detail(13, settings)
    legacy, compound = detail['sensory_inputs']
    assert not legacy['selected'] and compound['selected']
    assert compound['routing']['reason'] == 'spectral_channel_unavailable'
    assert compound['group'] is None
    # Reading under another profile does not mutate the cached routing.
    assert index.detail(13, SensorySettings())['sensory_inputs'][0]['selected']
    assert 'selected' not in index.inputs[2][0]
    assert index.detail(14, settings)['motor_output']['status'] == 'unmapped'
