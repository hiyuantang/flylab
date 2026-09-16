"""Full-graph FP16 optimization benchmark and exact state comparison.

Offline only: all imported neurons/edges, fixed 0.1 ms ticks, no live checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from flylab.full_connectome import FullBrain
from flylab.metal_dynamics import MetalDynamics, ActiveRowMetalDynamics, FusedActiveRowMetalDynamics, NEURAL_TENSORS


def digest(brain):
    result = hashlib.sha256()
    for name in NEURAL_TENSORS:
        result.update(name.encode())
        result.update(getattr(brain, name).cpu().numpy().tobytes())
    result.update(brain.generator.get_state().numpy().tobytes())
    result.update(str(brain.tick_index).encode())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/full'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/fp16-optimization.json'))
    parser.add_argument('--duration-ms', type=int, default=100)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--mark-lanes', type=int, choices=[1, 4, 8, 32], default=1)
    parser.add_argument('--fused', action='store_true', help='Compare fused active rows against the previous active-row implementation')
    args = parser.parse_args()
    if args.duration_ms < 1 or args.trials < 2:
        parser.error('Positive duration and at least two trials required')
    torch.set_num_threads(1)
    brain = FullBrain(args.data, device='mps')
    brain.dtype = torch.float16
    modes = {'baseline': MetalDynamics(brain.weights, weight_dtype=torch.float16, state_dtype=torch.float16),
             'active_rows': ActiveRowMetalDynamics(brain.weights, mark_lanes=1)}
    if args.fused:
        modes = {'baseline': modes['active_rows'], 'active_rows': FusedActiveRowMetalDynamics(brain.weights, mark_lanes=args.mark_lanes)}
        if args.output == Path('docs/results/fp16-optimization.json'):
            args.output = Path('docs/results/fp16-fused.json')
    elif args.mark_lanes != 1:
        modes = {'baseline': modes['active_rows'], 'active_rows': ActiveRowMetalDynamics(brain.weights, mark_lanes=args.mark_lanes)}
        if args.output == Path('docs/results/fp16-optimization.json'):
            args.output = Path(f'docs/results/fp16-mark{args.mark_lanes}.json')
    drive = torch.tensor([14.7 if r.get('class') == 'olfactory' or r.get('superclass') == 'ol_sensory' else 0.
                          for r in brain.neurons], dtype=torch.float16, device='mps')
    runs, snapshots = {}, {}
    for name, engine in modes.items():
        assert torch.equal(engine.ptr.cpu().long(), brain.weights.crow_indices())
        assert torch.equal(engine.pre.cpu().long(), brain.weights.col_indices())
        brain.metal = engine
        brain.reset()
        brain.advance(drive, 20.)
        runs[name] = {'kernel': engine.version, 'wall_seconds': [], 'spikes': [],
                      'gpu_connectivity_bytes': sum(t.numel()*t.element_size() for t in
                         [engine.ptr, engine.pre, engine.weight]) +
                         (sum(t.numel()*t.element_size() for t in [engine.out_ptr, engine.out_post, engine.active])
                          if isinstance(engine, ActiveRowMetalDynamics) else 0)}
        print(name, 'warmed', flush=True)
    for trial in range(args.trials):
        order = list(modes) if trial % 2 == 0 else list(reversed(modes))
        for name in order:
            brain.metal = modes[name]
            brain.reset()
            torch.mps.synchronize()
            started = time.perf_counter()
            brain.advance(drive, args.duration_ms)
            torch.mps.synchronize()
            elapsed = time.perf_counter()-started
            runs[name]['wall_seconds'].append(elapsed)
            runs[name]['spikes'].append(brain.total_spikes)
            print(f'trial {trial+1} {name}: {elapsed:.4f}s, {brain.total_spikes} spikes', flush=True)
    # Every stored neuron tensor, pending delay and integer state is hashed every
    # millisecond. This validation timing is excluded from performance results.
    traces, active_fraction = {}, []
    for name, engine in modes.items():
        brain.metal = engine
        brain.reset()
        trace = []
        for _ in range(args.duration_ms):
            brain.advance(drive, 1.)
            trace.append(digest(brain))
            if name == 'active_rows':
                active_fraction.append(float(engine.active.count_nonzero())/len(brain.ids))
        traces[name] = trace
        snapshots[name] = {key: getattr(brain, key).cpu().clone() for key in NEURAL_TENSORS}
        assert brain.tick_index == args.duration_ms*10
        assert brain.total_spikes == runs[name]['spikes'][0]
        assert all(torch.isfinite(t).all() for t in snapshots[name].values())
    exact = {key: torch.equal(snapshots['baseline'][key], snapshots['active_rows'][key]) for key in NEURAL_TENSORS}
    mismatch = [i+1 for i, (a, b) in enumerate(zip(traces['baseline'], traces['active_rows'])) if a != b]
    for values in runs.values():
        values['median_wall_seconds'] = statistics.median(values['wall_seconds'])
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
              'neurons': len(brain.ids), 'edges': brain.weights._nnz(),
              'graph_sha256': brain.manifest['graph_sha256'], 'connectivity_exact': True,
              'precision': 'float16', 'dt_ms': brain.config.dt_ms, 'duration_ms': args.duration_ms,
              'stimulus': '14.7 mV equivalent current to olfactory and ol_sensory neurons',
              'trials': args.trials, 'runs': runs,
              'comparison': 'active rows versus fused active rows' if args.fused else f'active rows versus {args.mark_lanes}-lane marking' if args.mark_lanes != 1 else 'full scan versus active rows',
              'speedup': runs['baseline']['median_wall_seconds']/runs['active_rows']['median_wall_seconds'],
              'endpoint_tensors_exact': exact, 'state_hash_interval_ms': 1,
              'different_state_hash_times_ms': mismatch, 'state_hashes': traces,
              'active_row_fraction_sampled_each_ms': {'mean': statistics.mean(active_fraction), 'max': max(active_fraction)},
              'limits': 'One 100 ms-style fixed stimulus from reset; exact sampled state agreement is not biological validation. Small tests additionally compare every tick. Includes host submission overhead; excludes construction, warmup, body simulation and state-comparison transfers. No live simulation changed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['speedup', 'endpoint_tensors_exact', 'different_state_hash_times_ms', 'active_row_fraction_sampled_each_ms']}, indent=2), flush=True)
    if mismatch or not all(exact.values()):
        raise SystemExit('Optimization changed FP16 state; do not use as an exact replacement')


if __name__ == '__main__':
    main()
