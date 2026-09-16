"""Compare full-history scratch allocation/reuse without changing live weights.

Run with --settings-from pointing to an existing checkpoint. No optimizer runs.
Both variants execute the same full graph, forward ticks and backward equations.
"""
import argparse
import json
import time
from pathlib import Path
from threading import Event
import numpy as np
import torch
from flylab.gesture_batch import BatchSession, demonstrations
from flylab.gesture_scene import random_placement
from flylab import gesture_history as history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-from', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.settings_from, map_location='cpu', weights_only=True)
    session = BatchSession(Path('data/full'), saved['settings'], saved['rank'], saved['seed'])
    targets, _ = demonstrations(session, args.steps, Event(), cues=['point_both'])
    cues = ['point_both'] * 3
    placements = [random_placement(np.random.default_rng(i)) for i in range(3)]
    allocate = history.window_buffers
    rows, expected = [], None
    # Warm up both tape shapes and compiled kernels, then interleave comparisons.
    for mode in ['warmup', 'allocate', 'reuse', 'reuse', 'allocate']:
        history.window_buffers = allocate if mode != 'allocate' else lambda engine, shape, reuse=False: allocate(engine, shape, False)
        session.gradient.parameters.grad = None
        torch.mps.synchronize()
        began = time.perf_counter()
        samples, _, _ = session.parallel_samples(cues, placements, targets, Event())
        norm, log_norm = history.clip_full_gradient(session.gradient)
        torch.mps.synchronize()
        seconds = time.perf_counter() - began
        actual = session.gradient.parameters.grad.detach().cpu().clone()
        if expected is None:
            expected = actual
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        row = {'mode': mode, 'seconds': seconds, 'losses': [s['loss'] for s in samples],
               'gradient_log10_norm': log_norm, 'gradient_identical': True}
        rows.append(row)
        print(json.dumps(row), flush=True)
    history.window_buffers = allocate
    args.output.write_text(json.dumps({'steps': args.steps, 'batch_size': 3, 'rows': rows,
        'scope': 'Full-sample BPTT; allocation versus scratch reuse only; no optimizer updates'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
