"""Compare isolated full-graph adapter training; never modify live/checkpoint state.

Run: PYTHONPATH=backend uv run python scripts/benchmark_gesture_training.py
"""
from dataclasses import asdict
from pathlib import Path
from threading import Event
import gc
import json
import time
import numpy as np
import torch
from flylab.body import BodyParameters
from flylab.gesture_batch import BatchSession, demonstrations
from flylab.gesture_scene import random_placement
from flylab.motor_mapping import EXTENDED_PROFILE
from flylab.senses import SensorySettings


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('This comparison requires an Apple GPU')
    results = []
    for device, precision in [('mps', 'float16'), ('cpu', 'float64')]:
        settings = {'body': asdict(BodyParameters(appendage_model='peripheral-v2', elasticity_profile='stance-elastic-v1')), 'mapping_profile': EXTENDED_PROFILE,
            'gains': [0.] * 8, 'senses': asdict(SensorySettings(vision_enabled=True, vision_model='compound-retina-v1')),
            'training_execution': {'device': device, 'precision': precision}}
        session = BatchSession(Path('data/full'), settings, 2, 42)
        targets, _ = demonstrations(session, 5, Event())
        initial = session.gradient.parameters.detach().clone()
        began = time.perf_counter()
        result, _, _ = session.sample('point', random_placement(np.random.default_rng(42)),
                                     targets['point'], Event(), 1.)
        from flylab.gesture_history import clip_full_gradient
        norm, log_norm = clip_full_gradient(session.gradient)
        torch.optim.Adam([session.gradient.parameters], lr=.01).step()
        if device == 'mps':
            torch.mps.synchronize()
        results.append({'device': device, 'precision': precision,
            'wall_seconds': time.perf_counter() - began, 'loss': result['loss'],
            'gradient_norm': norm, 'gradient_log10_norm': log_norm,
            'parameter_change': float((session.gradient.parameters.detach() - initial).norm()),
            'neurons': len(session.sim.full_brain.ids), 'edges': session.sim.full_brain.weights._nnz()})
        print(json.dumps(results[-1]), flush=True)
        del session, initial, targets
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    print(json.dumps({'simulated_seconds': .1, 'steps': 5, 'batch_size': 1, 'rank': 2,
        'seed': 42, 'results': results, 'speedup': results[1]['wall_seconds'] / results[0]['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
