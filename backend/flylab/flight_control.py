"""Declared engineering hover assist. No learned or connectome controller.

Only the six bounded wing servo targets are written. Body poses, velocities,
contacts and aerodynamic forces remain the responsibility of MuJoCo.
"""
import math

import mujoco
import numpy as np

from .flight import (TRIM_AMPLITUDE, TRIM_ELEVATION, TRIM_FREQUENCY,
                     TRIM_STROKE_BIAS)

CONTROLLER_VERSION = 'wing-hover-assist-v1'
SPINUP_SECONDS = .08


class HoverController:
    def __init__(self, rig):
        self.rig = rig
        self.target = np.array([0., 0., rig.settings.initial_height])
        # One beat of history rejects wingbeat-induced body vibration. A short
        # exponential filter alone feeds that vibration back into the pattern.
        self.history = np.zeros((round(1/(TRIM_FREQUENCY*rig.settings.timestep)), 12))
        self.total = np.zeros(12)
        self.cursor = 0
        self.height_integral = self.pitch_integral = 0.
        self.last_command = np.zeros(6)

    def command(self):
        r, d = self.rig, self.rig.data
        h = r.settings.timestep
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, d.qpos[3:7])
        rotation = rotation.reshape(3, 3)
        angles = [math.atan2(rotation[2, 1], rotation[2, 2]),
                  math.asin(float(np.clip(-rotation[2, 0], -1, 1))),
                  math.atan2(rotation[1, 0], rotation[0, 0])]
        observation = np.r_[d.qpos[:3]-self.target, d.qvel[:3], angles, d.qvel[3:6]]
        self.total += observation-self.history[self.cursor]
        self.history[self.cursor] = observation
        self.cursor = (self.cursor+1) % len(self.history)
        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = self.total/len(self.history)
        desired_pitch = np.clip(-.0008*x-.002*vx, -.15, .15)
        desired_roll = np.clip(.0008*y+.002*vy, -.15, .15)
        # Bounded integral terms accommodate small trim/model mismatch without
        # allowing unlimited integral accumulation when a motor reaches a limit.
        self.height_integral = float(np.clip(self.height_integral+z*h, -1, 1))
        self.pitch_integral = float(np.clip(self.pitch_integral+pitch*h, -1, 1))
        elevation = np.clip(TRIM_ELEVATION+.9*(pitch-desired_pitch)+.018*wy
                            +.2*self.pitch_integral, .1, .55)
        amplitude = np.clip(TRIM_AMPLITUDE-.04*z-.006*vz-.2*self.height_integral
                            +.6*(elevation-TRIM_ELEVATION), .85, 1.35)
        differential_amplitude = np.clip(-.08*(roll-desired_roll)-.002*wx, -.08, .08)
        differential_feather = np.clip(.2*yaw+.006*wz, -.3, .3)
        r.phase += 2*math.pi*TRIM_FREQUENCY*h
        stroke = math.sin(r.phase)
        feather = .8*math.tanh(3*math.cos(r.phase))
        self.last_command = np.array([
            elevation, TRIM_STROKE_BIAS+(amplitude+differential_amplitude)*stroke,
            feather+differential_feather,
            elevation, TRIM_STROKE_BIAS+(amplitude-differential_amplitude)*stroke,
            feather-differential_feather])
        return self.last_command


def free_rollout(rig, duration, *, assist=True, frequency=TRIM_FREQUENCY,
                 amplitude=TRIM_AMPLITUDE, drive=True, publish=None,
                 initial_rotation_vector=(0., 0., 0.), initial_velocity=(0., 0., 0.)):
    """Spin up on an explicit tether, then release without resetting the motion.

    This is an airborne release experiment, not a takeoff demonstration.
    """
    if not rig.data.eq_active[0]:
        raise ValueError('Free rollout requires the initial spin-up tether')
    h = rig.settings.timestep
    steps = round(duration/h)
    if steps < 1 or not math.isclose(steps*h, duration, abs_tol=1e-12):
        raise ValueError('Duration must contain a whole number of physics steps')
    if assist and (frequency != TRIM_FREQUENCY or amplitude != TRIM_AMPLITUDE or not drive):
        raise ValueError('Hover assist uses its calibrated wingbeat')
    rotation_vector, velocity = np.asarray(initial_rotation_vector), np.asarray(initial_velocity)
    if (rotation_vector.shape != (3,) or velocity.shape != (3,)
            or not np.isfinite(rotation_vector).all() or not np.isfinite(velocity).all()):
        raise ValueError('Initial perturbations must be finite three-vectors')
    rig.flap(SPINUP_SECONDS, frequency=frequency if drive else 0.,
             amplitude=amplitude if drive else 0., feather=.8 if drive else 0.)
    rig.data.eq_active[0] = False
    # Optional test initial conditions are applied once, at the release boundary.
    # No pose or velocity assignment takes place during the free rollout.
    angle = float(np.linalg.norm(rotation_vector))
    if angle:
        delta = np.r_[math.cos(angle/2), rotation_vector/angle*math.sin(angle/2)]
        quaternion = rig.data.qpos[3:7].copy()
        mujoco.mju_mulQuat(rig.data.qpos[3:7], delta, quaternion)
    if np.any(velocity):
        rig.data.qvel[:3] += velocity
    if angle or np.any(velocity):
        mujoco.mj_forward(rig.model, rig.data)
    start = float(rig.data.time)
    controller = HoverController(rig) if assist else None
    samples = []
    for index in range(steps):
        if controller:
            angles = controller.command()
        else:
            rig.phase += 2*math.pi*(frequency if drive else 0.)*h
            angles = [TRIM_ELEVATION, TRIM_STROKE_BIAS+(amplitude*math.sin(rig.phase) if drive else 0.),
                      .8*math.tanh(3*math.cos(rig.phase)) if drive else 0.]*2
        rig.advance(angles, h)
        sample = rig.measure()
        sample['physics_time_s'] = sample['time']
        sample['time'] -= start
        sample['force_time_s'] -= start
        samples.append(sample)
        if publish:
            publish(sample, index)
    return samples
