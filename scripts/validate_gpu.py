"""Compare full-graph CPU/MPS endpoints under identical deterministic drive.

Run: PYTHONPATH=backend uv run python scripts/validate_gpu.py
This short numerical check does not establish biological equivalence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import torch
from flylab.full_connectome import FullBrain

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=Path('data/full'))
parser.add_argument('--output', type=Path, default=Path('docs/results/mps-validation.json'))
parser.add_argument('--duration-ms', type=float, default=40.)
args = parser.parse_args()
torch.set_num_threads(1)
brain = FullBrain(args.data)
drive = torch.zeros(len(brain.ids), dtype=torch.float64)
for i, row in enumerate(brain.neurons):
    if row.get('class') == 'olfactory' or row.get('superclass') == 'ol_sensory':
        drive[i] = 14.7
runs, endpoints = {}, {}
for device in ['cpu', 'mps']:
    brain.set_device(device)
    brain.reset()
    brain.advance(drive, args.duration_ms)
    endpoints[device] = {key: getattr(brain, key).detach().cpu().clone()
                         for key in ['voltage', 'current', 'rates', 'spike_counts', 'delay_queue']}
    runs[device] = {**brain.execution_summary(), 'wall_seconds': brain.last_wall_seconds,
                    'spikes': brain.total_spikes, 'ticks': brain.tick_index}
    print(device, runs[device], flush=True)
errors = {}
for key in endpoints['cpu']:
    a, b = endpoints['cpu'][key], endpoints['mps'][key]
    errors[key] = {'max_absolute_error': float((a-b).abs().max()),
                   'mean_absolute_error': float((a.double()-b.double()).abs().mean()),
                   'different_elements': int((a != b).sum())}
result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
          'platform': platform.platform(), 'neurons': len(brain.ids), 'edges': brain.weights._nnz(),
          'gpu_edges': len(brain.metal.pre), 'graph_sha256': brain.manifest['graph_sha256'],
          'duration_ms': args.duration_ms, 'dt_ms': brain.config.dt_ms,
          'stimulus': '14.7 mV equivalent current to all olfactory-class and ol_sensory neurons; zero to others',
          'stimulated_neurons': int((drive != 0).sum()), 'runs': runs, 'endpoint_errors': errors,
          'limits': 'Endpoint comparison on one short fixed stimulus, not a full spike-train or biological fidelity validation. Float32 can change long-run trajectories.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(errors, indent=2), flush=True)
