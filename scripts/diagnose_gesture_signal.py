"""Trace full-brain visual latency and probe the measured surrogate direction.

Uses a saved run's configuration, but starts at the original adapter. Does not
modify live state or install weights. Longer traces hold the last short target
for diagnostic scoring only; they are not certified long-duration pose targets.
"""
import argparse
import json
from pathlib import Path
from threading import Event
import numpy as np
import torch
from flylab.gesture_batch import BatchSession, demonstrations
from flylab.gesture_scene import random_placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    session = BatchSession(args.graph, saved['settings'], saved['rank'], saved['seed'])
    retina = session.sim.bridge.retina()
    visual = torch.cat([group[0] for group in retina.groups])
    motor = torch.unique(torch.cat(session.sim.bridge.channels))
    targets, _ = demonstrations(session, 5, Event(), cues=['point_both'])
    prepare, advance = session.sim.bridge.prepare_command, session.gradient.advance
    records, current = {}, {}
    blank = False

    def prepare_trace(*values, **kwargs):
        drive, mask = prepare(*values, **kwargs)
        if blank:
            drive[visual] = 0
        current.setdefault('drive', []).append(drive.numpy().copy())
        return drive, mask

    def advance_trace(*values, **kwargs):
        rates = advance(*values, **kwargs)
        current.setdefault('rates', []).append(rates.detach().cpu().numpy().copy())
        return rates

    session.sim.bridge.prepare_command = prepare_trace
    session.gradient.advance = advance_trace
    report = {'configuration_source': str(args.checkpoint), 'starts_from': 'original adapter',
              'neural_dt_ms': .1, 'command_ms': 20, 'backprop_window_ms': 20,
              'neurons': len(session.sim.full_brain.ids), 'traces': {}}
    variants = [('both', 'point_both', 42, False, 5), ('palm', 'palm', 42, False, 5),
                ('jitter', 'point_both', 43, False, 5), ('blank', 'point_both', 42, True, 5),
                ('long_both', 'point_both', 42, False, 25), ('long_palm', 'palm', 42, False, 25)]
    for name, cue, seed, blank, steps in variants:
        current = {}
        target = targets['point_both']
        target = torch.cat([target, target[-1:].repeat(max(0, steps - 5), 1)])
        result, _, comparison = session.sample(cue, random_placement(np.random.default_rng(seed)),
                                               target, Event(), 0., train=False)
        records[name] = {k: np.stack(v) for k, v in current.items()}
        records[name]['actual'] = np.array(comparison['actual'])
        report['traces'][name] = {'loss': result['loss'],
            'visual_spiking': int((records[name]['rates'][-1, visual] > 0).sum()),
            'motor_spiking': int((records[name]['rates'][-1, motor] > 0).sum())}
        print(name, report['traces'][name], flush=True)
    report['short_comparisons'] = {name: {
        'drive_max_difference': float(np.max(abs(records[name]['drive'] - records['both']['drive']))),
        'motor_rate_max_difference': float(np.max(abs(records[name]['rates'][:, motor] - records['both']['rates'][:, motor]))),
        'activation_max_difference': float(np.max(abs(records[name]['actual'] - records['both']['actual'])))}
        for name in ('palm', 'jitter', 'blank')}
    a, b = records['long_both']['rates'], records['long_palm']['rates']
    report['motor_difference_per_20ms'] = np.max(abs(a[:, motor] - b[:, motor]), axis=1).tolist()
    report['first_difference_ms'] = {}
    for group in sorted({row.get('superclass') or 'unknown' for row in session.sim.full_brain.neurons}):
        ids = [i for i, row in enumerate(session.sim.full_brain.neurons) if (row.get('superclass') or 'unknown') == group]
        times = np.flatnonzero(np.max(abs(a[:, ids] - b[:, ids]), axis=1))
        report['first_difference_ms'][group] = int(20 * (times[0] + 1)) if len(times) else None
    gradients = {}
    place = random_placement(np.random.default_rng(42))
    for blank in (False, True):
        current = {}
        session.gradient.parameters.grad = None
        session.sample('point_both', place, targets['point_both'], Event(), 1., train=True)
        from flylab.gesture_history import clip_full_gradient
        norm, log_norm = clip_full_gradient(session.gradient)
        report['gradient_log10_norm_without_vision' if blank else 'gradient_log10_norm'] = log_norm
        gradients[blank] = session.gradient.parameters.grad.detach().clone()
    report['clipped_gradient_norm'] = float(gradients[False].norm())
    report['visual_gradient_max_difference'] = float((gradients[False] - gradients[True]).abs().max())
    blank = False
    base = session.gradient.parameters.detach().clone()
    direction = gradients[False] / gradients[False].norm()
    report['direction_probes'] = []
    for scale in (-.1, -.01, -.001, 0., .001, .01, .1):
        with torch.no_grad():
            session.gradient.parameters.copy_(base + scale * direction)
        current = {}
        result, _, _ = session.sample('point_both', place, targets['point_both'], Event(), 0., train=False)
        row = {'gradient_direction_displacement': scale, 'loss': result['loss']}
        report['direction_probes'].append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
