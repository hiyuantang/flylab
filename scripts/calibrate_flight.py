"""Fit three bounded wing kinematic parameters to static force/moment balance.

This calibrates the engineering rig internally. It does not fit experimental
Drosophila measurements or modify air properties, mass, geometry, or checkpoints.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from flylab.flight import FlightRig, FlightSettings, TRIM_FREQUENCY, average_wingbeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/results/wing-trim-calibration.json'))
    args = parser.parse_args()
    evaluations = []
    def residual(parameters):
        plane, elevation, amplitude = parameters
        rig = FlightRig(FlightSettings(stroke_plane_pitch=float(plane)))
        means = average_wingbeats(rig.flap(.08, elevation=float(elevation), amplitude=float(amplitude)), TRIM_FREQUENCY)
        error = np.array([means['mean_force_uN'][0], means['mean_force_uN'][2]-rig.mass_g*9810,
                          means['mean_torque_uN_mm'][1]])
        evaluations.append(dict(parameters=parameters.tolist(), residual=error.tolist()))
        print(len(evaluations), parameters.tolist(), error.tolist(), flush=True)
        return error
    parameters = np.array([.077, .29, 1.205])
    for _ in range(6):
        error = residual(parameters)
        if np.max(np.abs(error)) < .02:
            break
        increments = [.002, .005, .005]
        jacobian = np.column_stack([(residual(parameters+np.eye(3)[j]*step)-error)/step
                                    for j, step in enumerate(increments)])
        update = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
        parameters = np.clip(parameters+np.clip(update, -.03, .03), [-.1, .2, 1.1], [.2, .4, 1.3])
    error = residual(parameters)
    result = dict(parameters=dict(stroke_plane_pitch_rad=parameters[0], elevation_rad=parameters[1],
                                  stroke_amplitude_rad=parameters[2]),
                  residual_units=['uN horizontal', 'uN vertical minus weight', 'uN mm pitch about CoM'],
                  residual=error.tolist(), evaluations=evaluations,
                  accepted=bool(np.max(np.abs(error)) < .03),
                  interpretation='Internal static trim, not an empirical aerodynamic fit.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if not result['accepted']:
        raise SystemExit('Trim did not meet its acceptance threshold')


if __name__ == '__main__':
    main()
