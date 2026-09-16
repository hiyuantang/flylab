"""Compare explicit 0.1/1 ms clocks from reset without touching the live animal."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import torch
from flylab.full_connectome import FullBrain
from flylab.metal_dynamics import ActiveRowMetalDynamics, NEURAL_TENSORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=['float32', 'float16'], default='float32')
    parser.add_argument('--duration-ms', type=int, default=200)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    if args.duration_ms < 1 or args.trials < 2:
        parser.error('Positive duration and at least two trials required')
    torch.set_num_threads(1)
    brain = FullBrain(Path('data/full'), device='mps')
    if args.precision == 'float16':
        brain.metal = ActiveRowMetalDynamics(brain.weights)
        brain.dtype = torch.float16
    base = brain.config
    drive = torch.tensor([14.7 if r.get('class') == 'olfactory' or r.get('superclass') == 'ol_sensory' else 0.
                          for r in brain.neurons], dtype=brain.dtype, device='mps')

    def reset(dt):
        brain.config = replace(base, dt_ms=dt, timing_rounding='exact' if dt == .1 else 'ceil')
        brain.delay_steps = brain.config.integration_ticks(brain.config.delay_ms)
        brain.reset()

    runs = {dt: {'wall_seconds': [], 'spikes': []} for dt in [.1, 1.]}
    counts = {}
    for dt in runs:
        reset(dt)
        brain.advance(drive, 20.)
        torch.mps.synchronize()
    for trial in range(args.trials):
        for dt in ([.1, 1.] if trial % 2 == 0 else [1., .1]):
            reset(dt)
            torch.mps.synchronize()
            started = time.perf_counter()
            brain.advance(drive, args.duration_ms)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            assert brain.tick_index == round(args.duration_ms / dt)
            assert all(torch.isfinite(getattr(brain, name)).all() for name in NEURAL_TENSORS)
            if dt in counts:
                assert torch.equal(counts[dt], brain.spike_counts.cpu())
            counts[dt] = brain.spike_counts.cpu().clone()
            runs[dt]['wall_seconds'].append(elapsed)
            runs[dt]['spikes'].append(brain.total_spikes)
            print(f'{args.precision} trial {trial+1}, dt={dt}: {elapsed:.3f}s, {brain.total_spikes} spikes', flush=True)
    for dt, values in runs.items():
        values['median_wall_seconds'] = statistics.median(values['wall_seconds'])
        values['wall_seconds_per_simulated_second'] = values['median_wall_seconds'] * 1000 / args.duration_ms
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
              'precision': args.precision, 'kernel': brain.metal.version,
              'neurons': len(brain.ids), 'edges': brain.weights._nnz(), 'graph_sha256': brain.manifest['graph_sha256'],
              'duration_ms': args.duration_ms, 'trials': args.trials, 'runs': runs,
              'speedup': runs[.1]['median_wall_seconds'] / runs[1.]['median_wall_seconds'],
              'neurons_with_different_spike_counts': int((counts[.1] != counts[1.]).sum()),
              'max_spike_count_difference': int((counts[.1] - counts[1.]).abs().max()),
              'coarse_delay_ms': 2., 'coarse_refractory_ms': 3.,
              'limits': 'Fixed 14.7 drive to olfactory/ol_sensory neurons, from reset. Brain-only timing; excludes body, sensing and UI. Coarse timing changes dynamics; no equivalence claim. No live state modified.'}
    path = Path(f'docs/results/timestep-{args.precision}.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['speedup', 'neurons_with_different_spike_counts']}), flush=True)


if __name__ == '__main__':
    main()
