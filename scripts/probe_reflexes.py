"""Full-graph sensory pulse responses with a fixed body and matched background.

The body is deliberately held fixed for this diagnostic, not for locomotion.
Measures incremental knee torque predicted by the existing rate/muscle model.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from flylab.body import LEGS
from flylab.calibration import digest_array
from flylab.live_state import load_live
from calibrate_locomotion import file_hash, write_json


def assay(sim, progress=None, pulse_mV=14., vnc_log_gain=0.):
    if not np.isfinite(pulse_mV) or not 0 < pulse_mV <= 42:
        raise ValueError('Use a finite pulse current in (0, 42] mV')
    if not np.isfinite(vnc_log_gain) or not -2 <= vnc_log_gain <= 2:
        raise ValueError('VNC log gain must be finite and within [-2, 2]')
    sim.prepare_standing_trial()
    body, brain, bridge = sim.body, sim.full_brain, sim.bridge
    parameters = bridge.parameters.copy(); parameters[4] = vnc_log_gain
    bridge.set_parameters(parameters)
    frame = sim.sensory_frame()
    background, mask = bridge.prepare_command(body, sim.source, 0., frame=frame)
    parameters = bridge.parameters.copy()
    dt_ms = 1000/sim.command_hz
    warmup, pulse, tail = [round(v*sim.command_hz) for v in (.1, .05, .1)]
    receptor_indices = torch.cat(list(bridge.leg_feedback.groups.values())).to(brain.voltage.device)
    def trace(indices):
        brain.reset(); bridge.set_parameters(parameters)
        samples = []
        for i in range(warmup+pulse+tail):
            drive = background.clone()
            if warmup <= i < warmup+pulse:
                drive[indices] += pulse_mV
            brain.advance(drive, duration_ms=dt_ms, silence=mask)
            excitation = bridge.muscles()
            body.data.act[:] = excitation
            mujoco.mj_forward(body.model, body.data)
            samples.append({'time_ms': (i+1)*dt_ms,
                'receptor_rates_hz': brain.rates[receptor_indices].float().cpu().tolist(),
                'knee_torque_uN_mm': [float(body.data.qfrc_actuator[body.model.joint(leg.lower()+'_tibia_pitch').dofadr[0]]) for leg in LEGS],
                'excitation': excitation.tolist()})
        return samples
    control = trace(torch.tensor([], dtype=torch.long))
    baseline = np.array([s['knee_torque_uN_mm'] for s in control])
    results = []
    for (leg_index, channel), indices in bridge.leg_feedback.groups.items():
        ids = [int(brain.ids[i]) for i in indices]
        if not len(indices):
            results.append({'leg': LEGS[leg_index], 'channel': channel, 'body_ids': [], 'status': 'unmapped'})
            continue
        samples = trace(indices)
        delta = np.array([s['knee_torque_uN_mm'] for s in samples])-baseline
        # In the model's mirrored knee axes, positive torque flexes both sides.
        expected = -1 if channel in {'flexed', 'flexing'} else 1
        integral = delta[warmup:].sum(axis=0)*dt_ms/1000
        own = float(integral[leg_index])
        columns = [j for j, index in enumerate(receptor_indices.cpu().tolist()) if index in indices.tolist()]
        # A silent output is interpretable only if the intervention activated its targets.
        rates = np.array([s['receptor_rates_hz'] for s in samples])[:, columns]
        control_rates = np.array([s['receptor_rates_hz'] for s in control])[:, columns]
        rate_change = float(np.max((rates-control_rates)[warmup:].mean(axis=1)))
        result = {'leg': LEGS[leg_index], 'channel': channel, 'body_ids': ids,
            'peak_target_mean_rate_increase_hz': rate_change,
            'incremental_torque_integral_uN_mm_s': integral.tolist(),
            'predicted_restoring_torque_sign': expected,
            'matches_restoring_hypothesis': None if abs(own) < 1e-8 else bool(own*expected > 0),
            'status': ('targets_not_activated' if rate_change < 1e-5 else 'no_detectable_motor_increment') if abs(own) < 1e-8 else 'measured_model_response',
            'samples': samples}
        results.append(result)
        if progress:
            progress(f'{LEGS[leg_index]} {channel}: delta torque integral {own:.6g}, restoring={result["matches_restoring_hypothesis"]}')
    body.data.act[:] = 0
    mujoco.mj_forward(body.model, body.data)
    return {'protocol': 'fixed-body-reflex-v2', 'neurons': len(brain.ids), 'edges': brain.manifest['edges'],
        'graph_sha256': brain.manifest['graph_sha256'], 'execution': brain.execution_summary(),
        'body_pose_sha256': digest_array(body.data.qpos), 'sensory_background': frame['settings'],
        'warmup_ms': warmup*dt_ms, 'pulse_ms': pulse*dt_ms, 'tail_ms': tail*dt_ms,
        'pulse_current_mV': pulse_mV, 'bridge_parameters': parameters.tolist(),
        'receptor_body_ids': [int(brain.ids[i]) for i in receptor_indices.cpu().tolist()],
        'scope': 'All neurons and edges integrated; 50 ms receptor-group current pulse on identical fixed-pose sensory background. Model torque estimated at steady muscle activation, not a free-body or biological reflex measurement.',
        'control': control, 'results': results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    p.add_argument('--graph', type=Path, default=Path('data/full'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pulse-mv', type=float, default=14.)
    p.add_argument('--vnc-log-gain', type=float, default=0.)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    sim = load_live(args.checkpoint, args.graph)
    result = assay(sim, progress=lambda s: print(s, flush=True), pulse_mV=args.pulse_mv, vnc_log_gain=args.vnc_log_gain)
    result['script_sha256'] = file_hash(Path(__file__))
    result['checkpoint_sha256'] = file_hash(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({'restoring': sum(r.get('matches_restoring_hypothesis') is True for r in result['results']),
        'opposing': sum(r.get('matches_restoring_hypothesis') is False for r in result['results']),
        'untested_or_silent': sum(r.get('matches_restoring_hypothesis') is None for r in result['results'])}), flush=True)


if __name__ == '__main__':
    main()
