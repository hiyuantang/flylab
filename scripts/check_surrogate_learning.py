"""Isolated Adam check with fixed validation and randomized hand placement.

Starts from the original adapter and physically checked 500 ms targets. It never
installs the resulting weights or changes the live controller.
"""
import argparse
import json
from pathlib import Path
from threading import Event

import numpy as np
import torch

from flylab.gesture_batch import BatchSession, demonstrations
from flylab.gesture_history import clip_full_gradient
from flylab.gesture_scene import random_placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=.0003)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = {**saved['settings'], 'surrogate_scale': args.scale}
    session = BatchSession(Path('data/full'), config, 2, 42)
    engine = session.gradient
    targets, _ = demonstrations(session, 25, Event(), cues=['point_both'])
    target = targets['point_both']
    fixed = random_placement(np.random.default_rng(100))
    optimizer = torch.optim.Adam([engine.parameters], lr=.01)
    result, _, _ = session.sample('point_both', fixed, target, Event(), 0., train=False)
    baseline, rows = result['loss'], []
    for iteration in range(3):
        optimizer.zero_grad(set_to_none=True)
        placement = random_placement(np.random.default_rng(42 + iteration))
        result, _, _ = session.sample('point_both', placement, target, Event(), 1., train=True)
        norm, log_norm = clip_full_gradient(engine)
        optimizer.step()
        with torch.no_grad():
            engine.parameters.clamp_(-4, 4)
        validation, _, _ = session.sample('point_both', fixed, target, Event(), 0., train=False)
        row = {'iteration': iteration + 1, 'training_loss': result['loss'],
               'validation_loss': validation['loss'], 'gradient_norm': norm,
               'gradient_log10_norm': log_norm,
               'parameter_change': float((engine.parameters.detach().cpu() - torch.tensor(engine.adapter.initial)).norm())}
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps({'scale': args.scale, 'lr': .01, 'steps': 25,
            'batch_size': 1, 'iterations': 3, 'baseline_validation_loss': baseline, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
