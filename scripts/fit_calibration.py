"""Bounded reflex and muscle-strength calibration; never auto-installs a fit.

Fits qualitative restoring-response hypotheses and physical support, not
biological recordings. Retains every neuron/edge and every integration step.
"""
import argparse
import json
from pathlib import Path
import zipfile

from flylab.calibration import reflex_score, select_reflex_gain, run_trial, support_comparison
from flylab.live_state import load_live
from calibrate_locomotion import file_hash, write_json
from probe_reflexes import assay

GAINS = [-.5, 0., .15, .3, .5, 1.]
STRENGTHS = [.5, .75, 1., 1.25]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    p.add_argument('--graph', type=Path, default=Path('data/full'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output/'input.pt'
    checkpoint.write_bytes(args.checkpoint.read_bytes())
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'backend/flylab').glob('*.py')) + sorted((root/'scripts').glob('*.py'))
    protocol = {'version': 'bounded-calibration-v1', 'checkpoint_sha256': file_hash(checkpoint),
        'vnc_log_gains': GAINS, 'strength_scales': STRENGTHS,
        'source_sha256': {str(f.relative_to(root)): file_hash(f) for f in files},
        'scope': 'Exploratory parameter fitting against model hypotheses. No claim of measured biological calibration or preregistration.',
        'reflex_selection': 'Maximize restoring minus twice opposing counts on left legs; tie prefers unchanged gain.',
        'body_selection': 'Require nonzero activation, then maximize support fraction; tie minimizes loaded-foot travel.',
        'heldout': 'Right-leg reflexes; opposite push direction for body strength. No automatic promotion.'}
    write_json(args.output/'protocol.json', protocol)
    with zipfile.ZipFile(args.output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for f in files:
            archive.write(f, str(f.relative_to(root)))
    sim = load_live(checkpoint, args.graph)
    reports = []
    for gain in GAINS:
        result = assay(sim, vnc_log_gain=gain)
        write_json(args.output/f'reflex-vnc-{gain}.json', result)
        reports.append(result)
        print('VNC log gain', gain, reflex_score(result, 'L'), reflex_score(result, 'R'), flush=True)
    selected = reports[select_reflex_gain(reports)]
    gain = selected['bridge_parameters'][4]
    bodies = []
    for strength in STRENGTHS:
        result = run_trial(sim, vnc_log_gain=gain, strength_scale=strength,
                           progress=lambda s: print(s, flush=True))
        write_json(args.output/f'body-strength-{strength}.json', result)
        bodies.append(result)
    active = [b for b in bodies if b['metrics']['mean_muscle_activation'] > 1e-8]
    if not active:
        raise RuntimeError('All candidates are silent; refusing to label silence a calibration')
    best = max(active, key=lambda b: (b['metrics']['feet_supported_fraction'],
                                     -sum(b['metrics']['loaded_foot_travel_mm'])))
    strength = best['body_parameters']['strength_scale']
    heldout = {}; heldout_reports = {}
    for name, scale, condition in [('candidate', strength, 'intact'), ('baseline', 1., 'intact'), ('disconnected', strength, 'motor_disconnected')]:
        result = run_trial(sim, vnc_log_gain=gain if name == 'candidate' else 0.,
                           strength_scale=scale, condition=condition, push_uN=-3.,
                           progress=lambda s: print(s, flush=True))
        write_json(args.output/f'heldout-{name}.json', result)
        heldout[name] = result['metrics']
        heldout_reports[name] = result
    acceptance = support_comparison(heldout_reports['candidate'], heldout_reports['baseline'])
    baseline_reflex = reports[GAINS.index(0.)]
    right_not_worse = (reflex_score(selected, 'R')['score'] >= reflex_score(baseline_reflex, 'R')['score']
                       and reflex_score(selected, 'R')['opposing'] <= reflex_score(baseline_reflex, 'R')['opposing'])
    write_json(args.output/'fit.json', {'vnc_log_gain': gain, 'strength_scale': strength,
        'left_reflexes': reflex_score(selected, 'L'), 'right_reflexes': reflex_score(selected, 'R'),
        'heldout': heldout, 'applied_to_live': False,
        'acceptance': acceptance, 'right_reflexes_not_worse': right_not_worse,
        'passes_engineering_screen': acceptance['passes'] and acceptance['meaningful_improvement'] and right_not_worse,
        'interpretation': 'Candidate fit only. Compare with baseline and disconnected support; validate more conditions and biological data before interpreting the result as active balance.'})
    print(json.dumps({'vnc_log_gain': gain, 'strength_scale': strength, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
