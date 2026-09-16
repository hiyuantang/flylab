"""Offline full-graph FP16-weight experiment; never reads/writes live checkpoints.

Run: PYTHONPATH=backend uv run python scripts/benchmark_mixed_precision.py
Default GPU state/products/sums remain FP32; --all-fp16 adds a fully half
precision candidate. Timing excludes graph load/compilation;
validation runs separately with 1 ms spike-count bins. No body simulation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from flylab.full_connectome import FullBrain
from flylab.metal_dynamics import MetalDynamics


def compare(a, b):
    difference = (a.double() - b.double()).abs()
    return {'max_absolute_error': float(difference.max()),
            'mean_absolute_error': float(difference.mean()),
            'different_elements': int((a != b).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/full'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/mixed-precision.json'))
    parser.add_argument('--duration-ms', type=int, default=100)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--all-fp16', action='store_true', help='Also compare FP16 weights, state and arithmetic')
    args = parser.parse_args()
    if args.duration_ms < 1 or args.trials < 2:
        parser.error('Use a positive integer duration and at least two timing trials')
    torch.set_num_threads(1)
    brain = FullBrain(args.data)
    drive = torch.tensor([14.7 if r.get('class') == 'olfactory' or r.get('superclass') == 'ol_sensory' else 0.
                          for r in brain.neurons], dtype=torch.float64)
    keys = ['voltage', 'current', 'rates', 'spike_counts', 'delay_queue']
    brain.advance(drive, args.duration_ms)
    reference = {name: getattr(brain, name).cpu().clone() for name in keys}
    reference_wall = brain.last_wall_seconds
    brain.set_device('mps')
    modes = {'fp32': brain.metal, 'mixed': MetalDynamics(brain.weights, weight_dtype=torch.float16)}
    if args.all_fp16:
        modes['fp16'] = MetalDynamics(brain.weights, weight_dtype=torch.float16, state_dtype=torch.float16)
        if args.output == Path('docs/results/mixed-precision.json'):
            args.output = Path('docs/results/all-fp16.json')
    for mode, engine in modes.items():
        assert torch.equal(engine.pre.cpu().long(), brain.weights.col_indices())
        assert torch.equal(engine.ptr.cpu().long(), brain.weights.crow_indices())
        brain.metal = engine
        brain.dtype = engine.state_dtype
        brain.reset()
        brain.advance(drive, 20.)
        print(f'{mode}: warmed; {engine.weight.numel():,} edges retained', flush=True)
    measurements = {name: [] for name in modes}
    summaries = {}
    for trial in range(args.trials):
        # Rotate order to reduce a consistent cache/temperature/order advantage.
        names = list(modes)
        order = names[trial % len(names):] + names[:trial % len(names)]
        for mode in order:
            brain.metal = modes[mode]
            brain.dtype = modes[mode].state_dtype
            brain.reset()
            torch.mps.synchronize()
            start = time.perf_counter()
            brain.advance(drive, args.duration_ms)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            measurements[mode].append(elapsed)
            summaries[mode] = {**brain.execution_summary(), 'spikes': brain.total_spikes,
                               'ticks': brain.tick_index}
            print(f'trial {trial+1} {mode}: {elapsed:.4f}s, {brain.total_spikes:,} spikes', flush=True)
    endpoints, bins = {}, {}
    # Separate from speed measurement: compare spike counts in each 1 ms bin.
    for mode, engine in modes.items():
        brain.metal = engine
        brain.dtype = engine.state_dtype
        brain.reset()
        previous = torch.zeros(len(brain.ids), dtype=torch.int64)
        samples = []
        for _ in range(args.duration_ms):
            brain.advance(drive, 1.)
            counts = brain.spike_counts.cpu().clone()
            samples.append((counts - previous).numpy())
            previous = counts
        bins[mode] = np.stack(samples)
        endpoints[mode] = {name: getattr(brain, name).cpu().clone() for name in keys}
        assert brain.tick_index == round(args.duration_ms / brain.config.dt_ms)
        assert brain.total_spikes == summaries[mode]['spikes'], 'Batching changed total spike count'
        assert all(torch.isfinite(value).all() for value in endpoints[mode].values())
    w32 = modes['fp32'].weight.cpu()
    w16 = modes['mixed'].weight.cpu().float()
    def bin_comparison(mode):
        changed_bins = bins['fp32'] != bins[mode]
        first = np.flatnonzero(changed_bins.any(axis=1))
        return {'bin_ms': 1, 'different_neuron_bins': int(changed_bins.sum()),
                'neurons_with_any_difference': int(changed_bins.any(axis=0).sum()),
                'first_different_bin_start_ms': int(first[0]) if len(first) else None,
                'absolute_count_difference_sum': int(np.abs(bins['fp32'] - bins[mode]).sum())}
    medians = {name: statistics.median(times) for name, times in measurements.items()}
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
        'platform': platform.platform(), 'neurons': len(brain.ids),
        'edges': brain.weights._nnz(), 'graph_sha256': brain.manifest['graph_sha256'],
        'connectivity_exactly_preserved': True, 'duration_ms': args.duration_ms,
        'dt_ms': brain.config.dt_ms, 'stimulus': '14.7 mV equivalent current to olfactory-class and ol_sensory neurons',
        'stimulated_neurons': int((drive != 0).sum()), 'trials': args.trials,
        'timing_order': 'Rotating first mode each trial; all modes warmed for 20 ms first',
        'runs': {name: {**summaries[name], 'wall_seconds': measurements[name],
                        'median_wall_seconds': medians[name],
                        'simulated_per_wall_second': args.duration_ms / 1000. / medians[name],
                        'weight_bytes': modes[name].weight.numel() * modes[name].weight.element_size(),
                        'csr_bytes': sum(t.numel() * t.element_size() for t in
                                         [modes[name].ptr, modes[name].pre, modes[name].weight])}
                 for name in modes},
        'speedup_fp32_over_mixed': medians['fp32'] / medians['mixed'],
        'speedups': {mode: medians['fp32'] / medians[mode] for mode in modes},
        'weight_rounding': compare(w32, w16),
        'mixed_vs_fp32': {name: compare(endpoints['fp32'][name], endpoints['mixed'][name]) for name in keys},
        'spike_bins': bin_comparison('mixed'),
        'all_modes_vs_fp32': {mode: {'endpoint_errors': {name: compare(endpoints['fp32'][name], endpoints[mode][name]) for name in keys},
                                     'spike_bins': bin_comparison(mode)} for mode in modes if mode != 'fp32'},
        'cpu_fp64_reference': {'wall_seconds': reference_wall, 'spikes': int(reference['spike_counts'].sum()),
                               'gpu_endpoint_errors': {mode: {name: compare(reference[name], endpoints[mode][name])
                                                            for name in keys} for mode in modes}},
        'limits': 'One short fixed stimulus from reset, no body or long-run behavioral validation. 1 ms bins cannot detect all within-bin spike timing shifts. Timings depend on concurrent GPU load. Experimental precision modes are offline-only and cannot resume as an ordinary FP32 checkpoint. Live brain not changed.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'speedups': result['speedups'], 'spike_bins': {mode: row['spike_bins'] for mode, row in result['all_modes_vs_fp32'].items()},
                      'output': str(args.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
