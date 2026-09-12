"""Reproduce the complete motor inventory without loading or advancing a brain.

PYTHONPATH=backend uv run python scripts/audit_motor_mapping.py
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.feather as feather
import torch

from flylab.motor_mapping import PERIPHERAL_PROFILE, LEGACY_PROFILE, SOURCE
from flylab.neuromuscular import NeuromuscularBridge


def audit(directory):
    annotations = directory / 'neurons.feather'
    rows = feather.read_table(annotations).to_pylist()
    brain = SimpleNamespace(neurons=rows, ids=np.array([r['bodyId'] for r in rows]),
                            voltage=torch.zeros(len(rows), dtype=torch.float64))
    current = NeuromuscularBridge(brain, PERIPHERAL_PROFILE)
    legacy = NeuromuscularBridge(brain, LEGACY_PROFILE)
    manifest = json.loads((directory / 'manifest.json').read_text())
    return {'annotation_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
            'graph_sha256_from_manifest': manifest['graph_sha256'],
            'function_source': SOURCE,
            'scope': 'Annotation and routing audit; no dynamics or locomotion validation.',
            'legacy': {k: v for k, v in legacy.summary().items() if k not in {'mapping', 'unmapped'}},
            'current': current.summary()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('data/full'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/motor-mapping.json'))
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result['current'].items() if k not in {'mapping', 'unmapped'}}, indent=2))
