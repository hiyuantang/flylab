"""Audit external visual input routing without loading or advancing a brain.

PYTHONPATH=backend uv run python scripts/audit_retina.py
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pyarrow.feather as feather
from flylab.retina import RetinalRouting, eye_template, ASSET

if __name__ == '__main__':
    directory = Path('data/full')
    annotations = directory / 'neurons.feather'
    brain = SimpleNamespace(directory=directory, neurons=feather.read_table(annotations).to_pylist())
    routing = RetinalRouting(brain)
    report = {'model': 'compound-retina-v1',
              'scope': 'Input-assignment audit, not validation of cross-specimen anatomical registration or visual behavior.',
              'annotation_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
              'graph_sha256': json.loads((directory / 'manifest.json').read_text())['graph_sha256'],
              'optical_template_sha256': hashlib.sha256(ASSET.read_bytes()).hexdigest(),
              'source': {k:v for k,v in eye_template().items() if k not in {'left','right','right_hex'}},
              'summary': routing.summary, 'records': routing.records}
    Path('docs/results/retinal-routing.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(routing.summary,indent=2))
