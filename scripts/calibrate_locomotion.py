"""Reproducible offline motor, balance, and stepping experiments.

Reads a saved configuration but resets isolated trials; never saves over a life.
No policy or biological fidelity claim is produced by this exploratory assay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile

import mujoco
import torch

from flylab.calibration import CONDITIONS, motor_audit, feedback_audit, run_trial
from flylab.live_state import load_live, model_fingerprint


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    parser.add_argument('--output', type=Path, required=True, help='New directory; refuses to overwrite earlier experiments')
    parser.add_argument('--stage', choices=['audit', 'balance', 'stepping', 'all'], default='all')
    parser.add_argument('--seconds', type=float, default=4.)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 30:
        parser.error('Use 3 to 30 simulated seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    # Copy the input first so concurrent app saves cannot change this experiment.
    checkpoint = args.output/'input.pt'
    checkpoint.write_bytes(args.checkpoint.read_bytes())
    input_sha = file_hash(checkpoint)
    sim = load_live(checkpoint, args.graph)
    sim.prepare_standing_trial()
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'backend/flylab').glob('*.py')) + sorted((root/'scripts').glob('*.py'))
    metadata = {'protocol': 'locomotion-calibration-v1', 'checkpoint_sha256': input_sha,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': {str(p.relative_to(root)): file_hash(p) for p in files},
        'body_sha256': model_fingerprint(sim.body),
        'annotations_sha256': file_hash(args.graph/'neurons.feather'),
        'python': platform.python_version(), 'torch': torch.__version__, 'mujoco': mujoco.__version__,
        'stage': args.stage, 'seconds_per_trial': args.seconds,
        'scope': 'Exploratory single-model causal/calibration experiment, not biological validation or an independent-specimen statistical sample.'}
    write_json(args.output/'protocol.json', metadata)
    with zipfile.ZipFile(args.output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = str(path.relative_to(root))
            if file_hash(path) != metadata['source_sha256'][relative]:
                raise RuntimeError('Source changed while archiving the experiment')
            archive.write(path, relative)
        for relative in ['pyproject.toml', 'uv.lock', 'backend/flylab/assets/walking_reference.json',
                         'backend/flylab/assets/anatomy.json']:
            archive.write(root/relative, relative)
    # The archived input can be large and is ignored with the data directory.
    if args.stage in {'audit', 'all'}:
        write_json(args.output/'motor-audit.json', motor_audit(sim.body, sim.bridge))
        write_json(args.output/'feedback-audit.json', feedback_audit(sim.body, sim.bridge))
        print('Motor and feedback audits saved', flush=True)
    trials = []
    def run(name, **kwargs):
        report = run_trial(sim, seconds=args.seconds, progress=lambda s: print(s, flush=True), **kwargs)
        write_json(args.output/(name+'.json'), report)
        trials.append({'name': name, **{k: v for k, v in report.items() if k != 'samples'}})
        write_json(args.output/'summary.json', trials)
    if args.stage in {'balance', 'all'}:
        for condition in CONDITIONS:
            run('balance-'+condition, condition=condition)
        # Restricted calibration sweep: change only assumed sensory gain.
        for gain in [-2., -1.]:
            run('gain-'+str(int(gain)), sensory_log_gain=gain)
        # Opposite perturbation is held out from the gain ranking.
        candidates = [t for t in trials if t['condition'] == 'intact']
        best = min(candidates, key=lambda t: (-t['metrics']['post_push_supported_fraction'],
                                             t['metrics']['maximum_tilt_deg']))
        run('heldout-intact', push_uN=-3., sensory_log_gain=best['sensory_log_gain'])
        run('heldout-motor-disconnected', condition='motor_disconnected', push_uN=-3.)
        candidate, passive = trials[-2]['metrics'], trials[-1]['metrics']
        write_json(args.output/'selection.json', {'selection_rule': 'Highest post-push support fraction, then lowest peak tilt; training push +3 uN only.',
            'selected_sensory_log_gain': best['sensory_log_gain'], 'applied_to_live_app': False,
            'heldout_has_muscle_activity': candidate['mean_muscle_activation'] > 1e-8,
            'heldout_support_difference_from_passive': candidate['post_push_supported_fraction']-passive['post_push_supported_fraction'],
            'heldout_peak_tilt_difference_from_passive_deg': candidate['maximum_tilt_deg']-passive['maximum_tilt_deg'],
            'caution': 'A better standing score can result from suppressing neural output. Compare disconnected control before claiming active balance.'})
    if args.stage in {'stepping', 'all'}:
        run('step-no-command', push_uN=0.)
        for drive in [.4, 1.2, 2.4]:
            run('step-DNg100-'+str(drive), descending_drive=drive, push_uN=0.)
    print(f'Experiments saved to {args.output}; live checkpoint untouched.', flush=True)


if __name__ == '__main__':
    main()
