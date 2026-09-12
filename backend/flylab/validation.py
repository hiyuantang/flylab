"""Reproducible neural sensitivity probes and physical checkpoint evaluation.

Generated measurements characterize this model. They are not empirical fly data.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .full_connectome import FullBrain, Physiology, DYNAMICS_VERSION
from .neuromuscular import NeuromuscularBridge
from .physical_training import PhysicalTrainer, rollout
from .env import FlyEnv
from .body import BodyParameters
from .physical_policy import validate_policy
from .senses import SENSORY_VERSION


def neural_sweep(directory, gains=(2., 8., 20.), duration_ms=100):
    results = []
    for gain in gains:
        brain = FullBrain(directory, Physiology(gain=gain))
        bridge = NeuromuscularBridge(brain)
        brain.advance(torch.zeros(len(brain.ids)), duration_ms)
        spontaneous = brain.total_spikes
        drive = torch.zeros(len(brain.ids))
        sensory = torch.cat(bridge.olfactory)
        drive[sensory] = 3.
        brain.advance(drive, duration_ms)
        motor = bridge.regions['motor']
        results.append({'gain': gain, 'spontaneous_spikes': spontaneous,
                        'stimulated_population': 'all identified olfactory sensory neurons with L/R root side',
                        'stimulated_neurons': len(sensory), 'current': 3., 'duration_ms': duration_ms,
                        'total_spikes': brain.total_spikes, 'active_neurons': int((brain.rates > 1).sum()),
                        'active_motor_neurons': int((brain.rates[motor] > 1).sum()),
                        'max_motor_rate_hz': float(brain.rates[motor].max()),
                        'assumption': 'Gain sensitivity only; no experimental target or physiology fit.'})
    return {'dataset': brain.manifest['dataset'], 'graph_sha256': brain.manifest['graph_sha256'], 'results': results}


def evaluate_checkpoint(path, graph, horizon=250):
    trainer = PhysicalTrainer(path.parent, graph)
    checkpoint = trainer.load(path.name)
    parameters, _, _, environment = validate_policy(checkpoint)
    mode = checkpoint['mode']
    bridge = NeuromuscularBridge(FullBrain(graph, Physiology(**checkpoint['physiology']))) if mode == 'connectome' else None
    if bridge and bridge.brain.manifest['graph_sha256'] != checkpoint['graph_sha256']:
        raise ValueError('Graph/checkpoint mismatch')
    rows = []
    for mass, perturb in [(1., False), (1., True), (.9, False), (1.1, False)]:
        p = BodyParameters(**{**checkpoint['body_parameters'], 'mass_scale': checkpoint['body_parameters']['mass_scale'] * mass})
        env = FlyEnv(action_mode='muscle' if bridge else 'posture', task=checkpoint['task'], max_steps=horizon, body_parameters=p, scene_id=environment.scene_id)
        result = rollout(env, parameters, checkpoint['seed'] + 200, bridge, perturb=perturb, environment=environment)
        rows.append({'mass_multiplier': mass, **result})
    return {'checkpoint': path.name, 'horizon_steps': horizon, 'seconds_requested': horizon * .02, 'results': rows,
            'environment': environment.to_dict(), 'dynamics_version': DYNAMICS_VERSION if bridge else None,
            'sensory_version': SENSORY_VERSION,
            'checkpoint_dynamics_version': checkpoint.get('dynamics_version', 'lif-batch-rate-v1' if bridge else None),
            'interpretation': 'Held-out physical conditions; success here does not establish biological fidelity.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--horizon', type=int, default=250)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate_checkpoint(args.checkpoint, args.graph, args.horizon) if args.checkpoint else neural_sweep(args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
