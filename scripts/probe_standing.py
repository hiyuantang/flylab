"""Isolated standing validation; never writes the saved life or trains a policy.

Compare --mode neural (all imported neurons/edges) with --mode passive (body-only
control experiment). Optional external perturbation is a brief physical push,
never an upright assist. All sample times are simulated seconds.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from flylab.live_state import load_live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--mode', choices=['neural', 'passive'], default='neural')
    parser.add_argument('--push', action='store_true', help='3 µN lateral push for 0.1 s at simulated second 2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error('Duration must be finite and between 1 and 60 seconds')
    sim = load_live(args.checkpoint, args.graph)
    sim.prepare_standing_trial()
    body, brain = sim.body, sim.full_brain
    cycles = round(args.seconds * sim.command_hz)
    if not np.isclose(cycles / sim.command_hz, args.seconds):
        parser.error('Duration must align with the command clock')
    samples = []; start_position = body.data.qpos[:3].copy(); started = time.perf_counter()
    for i in range(cycles):
        body.data.xfrc_applied[body.body_ids['c_thorax'], 1] = 3. if args.push and 2 <= i / sim.command_hz < 2.1 else 0.
        if args.mode == 'neural':
            sim.advance()
        else:
            body.step_muscles(np.zeros(body.model.nu), dt=1/sim.command_hz)
        support = body.support_snapshot()
        support.update(time_s=float(body.data.time), position_mm=body.data.qpos[:3].tolist(),
            upright_cosine=float(body.data.xmat[body.body_ids['c_thorax']].reshape(3,3)[2,2]),
            speed_mm_s=float(np.linalg.norm(body.data.qvel[:3])),
            max_muscle_activation=float(body.data.act.max()))
        samples.append(support)
        if (i+1) % sim.command_hz == 0:
            print(f'{(i+1)/sim.command_hz:g}s: {support["supporting_feet"]} feet, {support["feet_support_fraction"]:.2f} body weights, {support["other_vertical_uN"]:.3f} µN other support', flush=True)
    settled = samples[round(.5*sim.command_hz):]
    report = {'mode': args.mode, 'push': args.push, 'simulated_seconds': args.seconds,
        'wall_seconds': time.perf_counter()-started, 'sample_hz': sim.command_hz,
        'graph_sha256': brain.manifest['graph_sha256'], 'neurons': len(brain.ids), 'edges': brain.manifest['edges'],
        'execution': brain.execution_summary(), 'body_parameters': body.mechanics()['parameters'],
        'motor_log_gain': float(sim.bridge.parameters[7]), 'sensory_settings': sim.sensory_frame()['settings'],
        'interpretation': 'Assumed elastic-supported stance. Full neural activity drives muscles in neural mode. No trained balance controller, gait tracker or biological standing validation.',
        'metrics_after_0_5s_settling': {
            'feet_supported_fraction': float(np.mean([s['feet_supported'] for s in settled])),
            'minimum_supporting_feet': min(s['supporting_feet'] for s in settled),
            'maximum_other_support_uN': max(s['other_vertical_uN'] for s in settled),
            'maximum_tilt_degrees': float(np.rad2deg(np.arccos(np.clip(min(s['upright_cosine'] for s in settled),-1,1)))),
            'net_xy_mm': float(np.linalg.norm(body.data.qpos[:2]-start_position[:2]))},
        'samples': samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['metrics_after_0_5s_settling']), flush=True)


if __name__ == '__main__':
    main()
