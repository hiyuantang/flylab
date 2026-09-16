"""Export physically replayable 50 Hz muscle commands and their activation labels."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from flylab.body import BodyParameters
from flylab.gesture_scene import GESTURES
from flylab.gesture_targets import muscle_reference, TARGET_VERSION

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--output', type=Path, default=Path('data/gesture-targets/current-body'))
args = parser.parse_args()
parameters = BodyParameters(appendage_model='peripheral-v2', elasticity_profile='stance-elastic-v1')
arrays, results = {}, {}
for cue in GESTURES:
    reference = muscle_reference(cue, args.steps, parameters)
    arrays[f'{cue}_commands'] = reference.commands
    arrays[f'{cue}_activations'] = reference.activations
    results[cue] = {**reference.evidence, 'frames': reference.metrics}
    print(cue, reference.evidence, flush=True)
arrays['actuator_names'] = np.array([reference.body.model.actuator(i).name for i in range(reference.body.model.nu)])
arrays['time_seconds'] = np.arange(1, args.steps + 1) * .02
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
report = {'target_version': TARGET_VERSION, 'body_parameters': asdict(parameters),
          'command_hz': 50, 'steps': args.steps, 'results': results,
          'all_verified': all(r['pose_reached'] for r in results.values()),
          'provenance': 'Calibrated engineering commands in the existing MuJoCo body; not biological recordings. Replay commands, not activations, from a fresh FlyBody with these exact parameters.'}
args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
if not report['all_verified']:
    raise SystemExit('At least one pose failed verification; inspect the report before use.')
