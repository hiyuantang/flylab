"""Isolated full-connectome locomotion trials; never modifies the saved life.

Reports physical movement and contact transitions, not a walking success claim.
Default LIF trials load the exact saved physiology. The explicit --rate option
tests an independent FP32 rate-model extension using measured neuron sizes.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from flylab.live_state import load_live
from flylab.senses import SensorSuite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    p.add_argument('--graph', type=Path, default=Path('data/full'))
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--target', choices=['DNg100', 'DNb08'], default='DNg100')
    p.add_argument('--drive', type=float, default=1.2, help='LIF threshold gaps; rate model arbitrary input units')
    p.add_argument('--feedback', choices=['off', 'legacy', 'opponent'], default='opponent')
    p.add_argument('--motor-log-gain', type=float, default=0.)
    p.add_argument('--rate', action='store_true', help='Explicitly use offline FP32 rate physiology, not live LIF')
    p.add_argument('--volumes', type=Path, default=Path('data/anatomy-reference/male-cns-volumes.npy'))
    p.add_argument('--volume-normalizer', type=float, default=None, help='Explicit normalization volume for rate-model comparisons; never changes graph selection')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 0 < args.seconds <= 30 or not 0 <= args.drive <= (1000 if args.rate else 3) or not -2 <= args.motor_log_gain <= 2:
        p.error('Invalid duration, drive or motor gain')
    sim = load_live(args.checkpoint, args.graph)
    sim.reset(); sim.odor = 'none'
    sim.sensors = SensorSuite(replace(sim.sensors.settings, vision_enabled=False, hearing_enabled=False,
        wind_enabled=False, touch_enabled=False, proprioception_enabled=args.feedback != 'off',
        proprioception_model='feco-opponent-v1' if args.feedback == 'opponent' else 'legacy-position-v1'))
    parameters = sim.bridge.parameters.copy(); parameters[7] = args.motor_log_gain
    sim.bridge.set_parameters(parameters)
    brain, bridge, body = sim.full_brain, sim.bridge, sim.body
    interval = 1 / sim.command_hz
    cycles = round(args.seconds / interval)
    if not np.isclose(cycles * interval, args.seconds): p.error('Duration must align with command clock')
    rates = None
    if args.rate:
        from flylab.cpg_rate import RateExperiment
        metadata = json.loads(args.volumes.with_suffix('.json').read_text())
        if metadata['body_ids_sha256'] != hashlib.sha256(brain.ids.tobytes()).hexdigest():
            raise ValueError('Volumes belong to different neuron IDs')
        if metadata['output_sha256'] != hashlib.sha256(args.volumes.read_bytes()).hexdigest():
            raise ValueError('Volume artifact checksum mismatch')
        rates = RateExperiment(brain, np.load(args.volumes), device=brain.device.type, dt_ms=1000/sim.command_hz/32, volume_normalizer=args.volume_normalizer)
    ids = bridge.locomotion.groups[args.target]
    if not len(ids): raise ValueError('Walking targets unavailable')
    samples = []
    # Constant input throughout trial; no periodic excitation or body target.
    if not rates:
        bridge.locomotion.submit(args.target, args.drive, min(5., args.seconds), sim.time)
    start = time.perf_counter()
    for i in range(cycles):
        if rates:
            drive = bridge.sensory_drive(body, sim.source, 0, frame=sim.sensory_frame())
            drive[ids] += args.drive
            rates.advance(drive, 1000/sim.command_hz)
            brain.rates = rates.rates
            # Same one-command delay used by the live pipelined workbench.
            command = bridge.muscles()
            body.step_muscles(sim.pending_command if sim.coupling_mode == 'pipelined' else command, dt=interval)
            sim.pending_command = command
            sim.time = (i+1)*interval
        else:
            if sim.time >= bridge.locomotion.expires:
                bridge.locomotion.submit(args.target, args.drive, min(5., args.seconds), sim.time)
            sim.advance()
        rotation = body.data.xmat[body.body_ids['c_thorax']].reshape(3,3)
        samples.append({'time': sim.time, 'position_mm': body.data.qpos[:3].tolist(),
            'upright_cosine': float(rotation[2,2]),
            'forward_mm_s': float(rotation[:,0] @ body.data.qvel[:3]),
            'foot_force_un': body.foot_feedback().tolist(),
            'muscle_excitation': body.data.ctrl[:84].tolist(),
            'circuit': bridge.locomotion.summary(sim.time)})
    pos = np.array([s['position_mm'] for s in samples]); up = np.array([s['upright_cosine'] for s in samples])
    contact = np.array([s['foot_force_un'] for s in samples]) > .1
    report = {'scope': 'Full graph; constant descending input; no gait tracker, direct motor stimulus, force assist or training.',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'graph_sha256': brain.manifest['graph_sha256'], 'neurons': len(brain.ids), 'edges': brain.manifest['edges'],
        'physiology': rates.metadata if rates else brain.execution_summary(),
        'feedback': bridge.leg_feedback.summary() if args.feedback == 'opponent' else args.feedback,
        'wall_seconds': time.perf_counter() - start, 'simulated_seconds': sim.time,
        'metrics': {'net_xy_mm': float(np.linalg.norm(pos[-1,:2]-pos[0,:2])),
            'forward_integral_mm': float(sum(s['forward_mm_s'] for s in samples)*interval),
            'upright_fraction_30deg': float(np.mean(up > np.cos(np.pi/6))),
            'air_to_contact_transitions': np.sum(contact[1:] & ~contact[:-1], axis=0).tolist(),
            'supported_fraction': float(np.mean(contact.sum(axis=1)>=3)),
            'walking_validated': False,
            'interpretation': 'Displacement and contact transitions alone cannot distinguish stepping from sliding or tipping.'},
        'samples': samples}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['neurons','edges','wall_seconds','simulated_seconds','metrics']},indent=2))


if __name__ == '__main__': main()
