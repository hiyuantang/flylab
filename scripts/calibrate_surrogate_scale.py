"""Compare backward surrogate scales without modifying live or saved weights.

The forward graph, precision, initial adapter, input and 500 ms target are fixed.
Each Adam trial starts from the original adapter with a fresh optimizer.
"""
import argparse
import json
import time
from pathlib import Path
from threading import Event

import numpy as np
import torch

from flylab.gesture_batch import BatchSession, demonstrations
from flylab.gesture_history import clip_full_gradient
from flylab.gesture_metal import training_kernels
from flylab.gesture_scene import random_placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', type=float, nargs='+', default=[1., .1, .03, .01, .003, .001, .0003])
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    session = BatchSession(Path('data/full'), saved['settings'], 2, 42)
    engine = session.gradient
    targets, _ = demonstrations(session, 25, Event(), cues=['point_both'])
    target = targets['point_both']
    placement = random_placement(np.random.default_rng(42))
    base = engine.parameters.detach().clone()
    rows = []
    expected_state = None
    for scale in args.scales:
        engine.lib = training_kernels(scale)
        engine.parameters.grad = None
        with torch.no_grad():
            engine.parameters.copy_(base)
        began = time.perf_counter()
        result, _, _ = session.sample('point_both', placement, target, Event(), 1., train=True)
        state = {name: getattr(engine.brain, name).detach().cpu().clone()
                 for name in ('voltage', 'current', 'rates', 'delay_queue', 'last_spike_tick', 'spike_counts')}
        if expected_state is None:
            expected_state = state
        for name, value in state.items():
            torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
        _, log_norm = clip_full_gradient(engine)
        gradient = engine.parameters.grad.detach().clone()
        energy = gradient.square().sum(dim=(0, 2)).cpu().numpy()
        row = {'scale': scale, 'forward_state_identical': True, 'baseline_loss': result['loss'], 'gradient_log10_norm': log_norm,
               'top_group_share': float(energy.max() / energy.sum()),
               'seconds': time.perf_counter() - began, 'adam_trials': []}
        for rate in [.001, .01]:
            with torch.no_grad():
                engine.parameters.copy_(base)
            engine.parameters.grad = gradient.clone()
            torch.optim.Adam([engine.parameters], lr=rate).step()
            result, _, _ = session.sample('point_both', placement, target, Event(), 0., train=False)
            row['adam_trials'].append({'lr': rate, 'loss': result['loss']})
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
