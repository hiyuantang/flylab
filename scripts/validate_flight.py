"""Validate the aerodynamic rig and its declared engineering hover controller."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from flylab.flight import (FlightRig, FlightSettings, TRIM_FREQUENCY, TRIM_AMPLITUDE,
                           TRIM_ELEVATION, TRIM_STROKE_BIAS, average_wingbeats)
from flylab.flight_control import CONTROLLER_VERSION, free_rollout


def run_case(case):
    name, mode, options, rotation, velocity = case
    settings = FlightSettings(**options)
    rig = FlightRig(settings)
    started = time.perf_counter()
    if mode in ('hover', 'fixed'):
        duration = .08 if settings.air_density == 0 else (1. if mode == 'hover' else .2)
        samples = free_rollout(rig, duration, assist=mode == 'hover',
                               initial_rotation_vector=rotation, initial_velocity=velocity)
    else:
        duration = .08
        samples = rig.flap(duration, frequency=0. if mode == 'still' else TRIM_FREQUENCY,
                           amplitude=0. if mode == 'still' else TRIM_AMPLITUDE,
                           feather=0. if mode == 'still' else .8)
    means = average_wingbeats(samples, 0. if mode == 'still' else TRIM_FREQUENCY)
    max_tilt = math.degrees(math.acos(float(np.clip(min(s['upright'] for s in samples), -1, 1))))
    height = max(abs(s['position_mm'][2]-settings.initial_height) for s in samples)
    drift = max(float(np.linalg.norm(s['position_mm'][:2])) for s in samples)
    environment_contacts = max(s['environment_contacts'] for s in samples)
    result = dict(settings=asdict(settings), mode=mode, **means, final=samples[-1],
        lift_weight_ratio=means['mean_force_uN'][2]/(rig.mass_g*9810),
        max_height_error_mm=height, max_tilt_degrees=max_tilt, max_horizontal_drift_mm=drift,
        environment_contacts=environment_contacts, max_wing_contacts=max(s['wing_contacts'] for s in samples),
        wing_contact_fraction=float(np.mean([s['wing_contacts'] > 0 for s in samples])),
        initial_rotation_vector_rad=rotation, initial_velocity_mm_s=velocity,
        wall_seconds=time.perf_counter()-started, physics_wall_seconds=rig.elapsed_wall,
        simulated_seconds=duration, spinup_seconds=.08 if mode in ('hover', 'fixed') else 0.,
        physics_steps=len(samples)+(round(.08/settings.timestep) if mode in ('hover', 'fixed') else 0),
        no_applied_body_force=not bool(np.any(rig.data.xfrc_applied) or np.any(rig.data.qfrc_applied)),
        flight_check_passed=(max_tilt <= 15 and height <= 1 and drift <= 5 and environment_contacts == 0)
            if mode in ('hover', 'fixed') else None)
    print(name, 'lift/weight', round(result['lift_weight_ratio'], 3),
          'tilt', round(max_tilt, 2), 'height error', round(height, 3), flush=True)
    return name, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/results/wing-fluid-validation.json'))
    parser.add_argument('--full', action='store_true', help='Include seven additional one-second robustness cases')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    zero = [0., 0., 0.]
    cases = [(name, mode, options, zero, zero) for name, mode, options in [
        ('tethered_air', 'tether', {}),
        ('tethered_vacuum', 'tether', dict(air_density=0., air_viscosity=0.)),
        ('stationary', 'still', {}),
        ('half_step', 'tether', dict(timestep=.000025)),
        ('quarter_step', 'tether', dict(timestep=.0000125)),
        ('fixed_free', 'fixed', {}),
        ('hover', 'hover', {}),
        ('hover_vacuum', 'hover', dict(air_density=0., air_viscosity=0.)),
    ]]
    if args.full:
        a = math.radians(5)
        cases += [(name, 'hover', options, rotation, velocity) for name, options, rotation, velocity in [
            ('hover_half_step', dict(timestep=.000025), zero, zero),
            ('pitch_positive', {}, [0, a, 0], zero), ('pitch_negative', {}, [0, -a, 0], zero),
            ('roll_positive', {}, [a, 0, 0], zero), ('roll_negative', {}, [-a, 0, 0], zero),
            ('velocity_positive', {}, zero, [25, 0, 0]), ('velocity_negative', {}, zero, [-25, 0, 0]),
        ]]
    if args.workers == 1:
        results = dict(map(run_case, cases))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            results = dict(pool.map(run_case, cases))
    air = results['tethered_air']
    checks = dict(
        trim_force=abs(air['lift_weight_ratio']-1) < .01,
        trim_pitch_moment=abs(air['mean_torque_uN_mm'][1]) < .03,
        vacuum_has_no_air_force=results['tethered_vacuum']['mean_force_uN'] == [0., 0., 0.],
        still_wings_have_no_hover_lift=abs(results['stationary']['lift_weight_ratio']) < .01,
        vacuum_cannot_hover=not results['hover_vacuum']['flight_check_passed'],
        timestep_refinement=all(abs(results[n]['lift_weight_ratio']/air['lift_weight_ratio']-1) < .03
                                for n in ['half_step', 'quarter_step']),
        all_air_hover_trials_pass=all(r['flight_check_passed'] for r in results.values()
                                     if r['mode'] == 'hover' and r['settings']['air_density'] > 0),
        no_hidden_support=all(r['no_applied_body_force'] for r in results.values()))
    payload = dict(interpretation='Model-specific calibration and one-second engineering hover checks; '
                   'not experimental biological validation, ground takeoff, or a learned flight policy.',
        model=FlightRig().description(), controller=CONTROLLER_VERSION, workers=args.workers,
        platform=platform.platform(),
        pattern=dict(frequency_hz=TRIM_FREQUENCY, stroke_amplitude_rad=TRIM_AMPLITUDE,
                     elevation_rad=TRIM_ELEVATION, stroke_bias_rad=TRIM_STROKE_BIAS, feather_amplitude_rad=.8),
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in ['backend/flylab/flight.py', 'backend/flylab/flight_control.py']},
        criteria=dict(max_height_error_mm=1, max_tilt_degrees=15, max_horizontal_drift_mm=5,
                      environment_contacts=0), checks=checks, cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    if not all(checks.values()):
        raise SystemExit('Flight validation failed: '+str([k for k, v in checks.items() if not v]))


if __name__ == '__main__':
    main()
