"""Check every imported neuron against source fields and generate all evidence records."""
import json
import gzip
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pyarrow.feather as feather
import torch

from flylab.full_connectome import Physiology
from flylab.neuromuscular import NeuromuscularBridge
from flylab.motor_mapping import EXTENDED_PROFILE
from flylab.neuron_evidence import NeuronEvidence, clean
from flylab.senses import SensorySettings


def main():
    started = time.perf_counter()
    directory = Path('data/full')
    rows = feather.read_table(directory / 'neurons.feather').to_pylist()
    source = {r['bodyId']: r for r in feather.read_table('data/body-annotations.feather').to_pylist()}
    transmitters = {r['body']: r for r in feather.read_table('data/body-neurotransmitters.feather').to_pylist()}
    brain = SimpleNamespace(neurons=rows, lookup={r['bodyId']: i for i, r in enumerate(rows)},
        ids=np.array([r['bodyId'] for r in rows]), voltage=torch.zeros(len(rows)), directory=directory,
        config=Physiology.paper(), manifest=json.loads((directory/'manifest.json').read_text()))
    bridge = NeuromuscularBridge(brain, EXTENDED_PROFILE)
    evidence = NeuronEvidence(brain, bridge)
    settings = SensorySettings(vision_model='compound-retina-v1', proprioception_model='feco-opponent-v1')
    claims = Counter()
    export = Path('data/neuron-evidence.jsonl.gz')
    with gzip.open(export, 'wt', compresslevel=6) as stream:
        for row in rows:
            body_id = row['bodyId']
            original = source[body_id]
            for key, value in original.items():
                assert clean(row[key]) == clean(value), (body_id, key)
            nt = transmitters.get(body_id, {})
            assert clean(row['transmitter']) == clean(nt.get('consensus_nt'))
            assert clean(row['transmitter_confidence']) == clean(nt.get('predicted_nt_confidence'))
            record = evidence.detail(body_id, settings)
            assert record['body_id'] == body_id and len(record['claims']) == 5
            assert record['sources'] and record['physiology']
            claims.update(c['level'] for c in record['claims'])
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    summary = evidence.metadata()
    summary.update(all_source_annotation_fields_match=True, all_transmitter_fields_match=True,
        records_checked=len(rows), claim_counts=dict(claims), export=str(export), export_bytes=export.stat().st_size,
        audit_seconds=time.perf_counter()-started,
        note='Read-only audit of installed graph/source tables. No neural simulation or live state changes. Export uses the stated audit profiles; selected sensor switches are not a live snapshot.')
    Path('docs/results/neuron-evidence.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ('total','sensory_total','motor_total','missing_type','missing_transmitter','missing_transmitter_score','all_source_annotation_fields_match','all_transmitter_fields_match','export_bytes','audit_seconds')}, indent=2))


if __name__ == '__main__':
    main()
