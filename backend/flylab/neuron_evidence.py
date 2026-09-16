"""Read-only, claim-specific provenance for every imported neuron.

Dataset support is provenance, not a probability or functional validation.
Descriptions are generated from retained fields and implemented input groups.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import math

SOURCE = 'https://male-cns.janelia.org/download/'
FIELDS = ('bodyId', 'type', 'instance', 'superclass', 'class', 'subclass',
          'somaSide', 'rootSide', 'entryNerve', 'mancType', 'transmitter')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class NeuronEvidence:
    def __init__(self, brain, bridge):
        self.brain, self.bridge = brain, bridge
        self.search = [' '.join(str(r.get(k) or '') for k in FIELDS).lower() for r in brain.neurons]
        self.motors = {r['body_id']: r for r in bridge.motor_inventory}
        self.inputs = defaultdict(list)
        for attr, description in {
            'olfactory': 'Odor concentration → assumed generic olfactory current; odor identities share tuning.',
            'visual': 'Legacy vision: mean brightness pooled per eye → assumed current.',
            'auditory': 'Sound RMS at the antenna → assumed frequency tuning and current gain.',
            'wind_gravity': 'Wind and head tilt → assumed antennal tuning and current gain.',
            'touch': 'Leg contact → assumed normalized tactile current.',
            'position': 'Legacy proprioception: joint angle → assumed generic position current.',
        }.items():
            for group, ids in enumerate(getattr(bridge, attr)):
                for i in ids.tolist():
                    self.inputs[int(i)].append({'mode': attr, 'group': group, 'basis': description})
        for (leg, channel), ids in bridge.leg_feedback.groups.items():
            for i in ids.tolist():
                self.inputs[int(i)].append({'mode': 'feco', 'group': leg,
                    'basis': f'FeCO opponent feedback: {channel}; inferred tuning and uncalibrated angle/velocity curves.',
                    'source': 'https://elifesciences.org/reviewed-preprints/97766v1'})
        self.sensory = {i for i, r in enumerate(brain.neurons)
                        if 'sensory' in str(r.get('superclass') or '') or i in self.inputs}
        self.retinal = None

    def metadata(self):
        rows = self.brain.neurons
        return {'version': 'neuron-evidence-v1', 'total': len(rows),
                'sensory_total': len(self.sensory), 'motor_total': len(self.motors),
                'classes': dict(sorted(Counter(r.get('superclass') or 'unspecified' for r in rows).items())),
                'missing_type': sum(not r.get('type') for r in rows),
                'missing_transmitter': sum(not r.get('transmitter') for r in rows),
                'missing_transmitter_score': sum(clean(r.get('transmitter_confidence')) is None for r in rows),
                'manifest': self.brain.manifest,
                'scope': 'All imported neurons. Field-based descriptions; not an individual literature review of every neuron. Excluded annotation rows remain outside this inventory.',
                'confidence_policy': 'Dataset-supported anatomy; source annotations; predicted transmitters; assumed dynamics; separately assessed peripheral routes. No overall confidence percentage.'}

    def item(self, i):
        row = self.brain.neurons[i]
        return {'body_id': int(row['bodyId']), 'type': row.get('type'),
                'superclass': row.get('superclass'), 'class': row.get('class'),
                'transmitter': row.get('transmitter'),
                'transmitter_score': clean(row.get('transmitter_confidence')),
                'sensory': i in self.sensory, 'motor': int(row['bodyId']) in self.motors}

    def page(self, query='', superclass='', scope='all', offset=0, limit=30):
        query = query.strip().lower()
        indices = [i for i, r in enumerate(self.brain.neurons)
                   if (not superclass or r.get('superclass') == superclass)
                   and (scope != 'sensory' or i in self.sensory)
                   and (scope != 'motor' or int(r['bodyId']) in self.motors)
                   and (not query or query in self.search[i])]
        return {'total': len(indices), 'offset': offset, 'limit': limit,
                'records': [self.item(i) for i in indices[offset:offset+limit]]}

    def detail(self, body_id, settings):
        i = self.brain.lookup.get(body_id)
        if i is None:
            raise ValueError('Neuron ID is outside the imported graph')
        row = self.brain.neurons[i]
        inputs = list(self.inputs.get(i, []))
        inputs = [dict(entry) for entry in inputs]
        for entry in inputs:
            mode = entry['mode']
            selected = (mode != 'visual' or settings.vision_model == 'legacy-grid-v2') and (
                mode != 'position' or settings.proprioception_model == 'legacy-position-v1') and (
                mode != 'feco' or settings.proprioception_model == 'feco-opponent-v1')
            flag = {'visual': 'vision_enabled', 'auditory': 'hearing_enabled',
                    'wind_gravity': 'wind_enabled', 'touch': 'touch_enabled',
                    'position': 'proprioception_enabled', 'feco': 'proprioception_enabled'}.get(mode)
            entry.update(selected=selected, enabled=bool(getattr(settings, flag)) if flag else None,
                         confidence='assumed encoding')
        if row.get('superclass') == 'ol_sensory' and row.get('class') == 'visual':
            if self.retinal is None:
                self.retinal = {r['body_id']: r for r in self.bridge.retina().records}
            record = self.retinal.get(body_id)
            if record:
                inputs.append({'mode': 'compound vision', 'group': record.get('facet'),
                    'basis': 'Column inferred from measured connectivity; optical registration across specimens and RGB tuning are unvalidated.',
                    'source': 'https://doi.org/10.1038/s41586-025-09276-5',
                    'selected': settings.vision_model == 'compound-retina-v1',
                    'enabled': settings.vision_enabled, 'confidence': 'inferred route; assumed encoding',
                    'routing': record})
        nt = row.get('transmitter')
        config = asdict(self.brain.config)
        sign = {'acetylcholine': 1., 'ach': 1., 'gaba': -1.,
                'histamine': config['histamine_sign'], 'glutamate': config['glutamate_sign'],
                'glut': config['glutamate_sign']}.get(str(nt).lower(), config['unknown_sign'])
        return clean({**self.item(i), 'annotations': row,
            'claims': [
                {'name': 'Identity and classification', 'level': 'source annotation',
                 'basis': 'MaleCNS curated fields reproduced below. Missing fields remain unknown; no per-neuron identity probability is supplied by this import.'},
                {'name': 'Neural wiring', 'level': 'dataset-supported anatomy',
                 'basis': 'All retained directed pairs and integer synapse counts come from the MaleCNS minconf-0.5 release. This threshold is not a confidence score for each neuron or connection. Individual synapse confidence values are not retained in the aggregate graph.'},
                {'name': 'Transmitter', 'level': 'source prediction' if nt else 'unresolved',
                 'basis': 'consensus_nt and predicted_nt_confidence from the source transmitter table. The score concerns that prediction, not neuron function or overall model accuracy.'},
                {'name': 'Electrical behavior', 'level': 'model assumption',
                 'basis': f'Current configured fast-synaptic sign: {sign:g}. Equations, gain, delay, threshold and refractory timing are model choices; anatomical counts are not measured electrical efficacy. A zero sign preserves the anatomical edge but supplies no fast current.'},
                {'name': 'Behavioral function', 'level': 'not independently assessed',
                 'basis': 'The annotated class/type describes this record. This inventory does not establish an experimentally validated behavioral role for each neuron.'},
            ], 'sources': [{'title': 'MaleCNS source documentation', 'url': SOURCE},
                          *[{'title': s['file'], **s} for s in self.brain.manifest.get('sources', [])]],
            'sensory_inputs': inputs,
            'sensory_explanation': ('Input routes are listed with their selected profile and modality switch. Enabled does not mean a nonzero stimulus; odor also depends on experiment settings.' if inputs else
                'No implemented external sensory route is identified. Imported neural connections remain present.' if i in self.sensory else
                'No direct sensory interface assigned; this neuron participates through its imported neural connections.'),
            'motor_output': self.motors.get(body_id), 'physiology': config,
            'intervention': ('This type is available for an experimental descending current pulse; pulse amplitude and duration are user/model choices.'
                             if row.get('type') in {'DNg100', 'DNb08'} else
                             'Workbench interventions can alter neural drive or silence neurons; these are experimental controls, not source wiring.')})
