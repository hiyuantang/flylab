"""Physically checked engineering demonstrations at the actual 50 Hz command rate.

These are simulated effective-muscle targets, not recorded biological activations.
The planner never controls the neural fly. Its commands are replayed in an isolated,
free body; labels come from that replay, including the stance-support muscles.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import mujoco
import numpy as np

from .body import BodyParameters, FlyBody, MAX_FORCE, MOMENT_ARM
from .gesture_scene import GESTURES, RAISED_FRONT_LEGS

TARGET_VERSION = 'routed-muscle-reference-v3'
COMMAND_DT = .02
PLANNER_DT = .001


def default_reference_channels(parameters):
    from .motor_mapping import EXTENDED_PROFILE, PERIPHERAL_PROFILE, PRETARSAL_PROFILE, MAPPING_PROFILE
    profile = {'peripheral-v2': EXTENDED_PROFILE, 'peripheral-v1': PERIPHERAL_PROFILE,
               'pretarsal-v1': PRETARSAL_PROFILE}.get(parameters.appendage_model, MAPPING_PROFILE)
    data = json.loads((Path(__file__).parent / 'assets/gesture-target-routes.json').read_text())
    return tuple(data['profiles'][profile])


@lru_cache(maxsize=16)
def reference_geometry(parameters, available_channels=None):
    available_channels = default_reference_channels(parameters) if available_channels is None else available_channels
    movable = np.zeros(42, dtype=bool)
    movable[np.asarray(available_channels, dtype=int) // 2] = True
    body = FlyBody(parameters)
    initial = body.data.qpos.copy()
    feet = body.data.site_xpos[body.foot_ids].copy()
    limits = body.model.jnt_range[body.active_ids]
    targets = {}
    for cue in GESTURES:
        data = mujoco.MjData(body.model)
        data.qpos[:] = initial
        desired = feet.copy()
        desired[:, 2] = .03  # All support feet start on the same ground plane.
        if cue == 'point_both' and len(available_channels) < 84:
            data.qpos[0] -= .1  # Shift support under the body before lifting both forelegs.
        if cue == 'fist':
            data.qpos[2] -= .25
        for leg in RAISED_FRONT_LEGS.get(cue, ()):
            desired[['LF', 'LM', 'LH', 'RF', 'RM', 'RH'].index(leg), 2] = 1.2 if cue == 'point_both' else .8
        for _ in range(160):
            mujoco.mj_forward(body.model, data)
            error = desired - data.site_xpos[body.foot_ids]
            for leg, site in enumerate(body.foot_ids):
                columns = slice(leg * 7, (leg + 1) * 7)
                jac = np.zeros((3, body.model.nv))
                mujoco.mj_jacSite(body.model, data, jac, None, site)
                local = jac[:, body.vadr[columns]].copy()
                local[:, ~movable[columns]] = 0
                change = local.T @ np.linalg.solve(local @ local.T + np.eye(3) * .003, error[leg])
                q = body.qadr[columns]
                data.qpos[q] = np.clip(data.qpos[q] + np.clip(change, -.08, .08),
                                      limits[columns, 0], limits[columns, 1])
        mujoco.mj_forward(body.model, data)
        residual = float(np.linalg.norm(desired - data.site_xpos[body.foot_ids], axis=1).max())
        if residual > .05:
            raise ValueError(f'{cue}: cannot construct a grounded reference (IK error {residual:.3f} mm)')
        targets[cue] = data.qpos[body.qadr].copy()
    return targets, float(initial[2])


def evaluate_reference(body, cue, initial_height):
    """Measure actual free-body movement; no IK residual is counted as success."""
    heights = body.data.site_xpos[body.foot_ids, 2]
    raised = [i for i, leg in enumerate(['LF', 'LM', 'LH', 'RF', 'RM', 'RH'])
              if leg in RAISED_FRONT_LEGS.get(cue, ())]
    supporting = [i for i in range(6) if i not in raised]
    upright = float(body.data.xmat[body.body_ids['c_thorax']].reshape(3, 3)[2, 2])
    height = float(body.data.qpos[2])
    supported = bool(np.all(heights[supporting] < .12) and np.all(heights > -.02))
    reached = supported and upright > .95 and height > .5
    if raised:
        reached = reached and bool(np.all(heights[raised] >= .3))
    elif cue == 'fist':
        reached = reached and height <= initial_height - .20
    else:
        reached = reached and height >= initial_height - .15
    return {'pose_reached': bool(reached), 'foot_heights_mm': heights.tolist(),
            'raised_legs': raised, 'support_feet_near_floor': supported,
            'body_height_mm': height, 'upright': upright}


@dataclass
class ReferenceRollout:
    body: FlyBody
    activations: np.ndarray
    commands: np.ndarray
    metrics: list[dict]
    evidence: dict


def muscle_reference(cue, steps, parameters=BodyParameters(), stop=None, available_channels=None):
    if cue not in GESTURES or not isinstance(steps, int) or not 1 <= steps <= 250:
        raise ValueError('Expected a gesture and 1–250 reference steps')
    available_channels = default_reference_channels(parameters) if available_channels is None else tuple(available_channels)
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < 84 for i in available_channels):
        raise ValueError('Invalid reference muscle channels')
    allowed = np.zeros(84, dtype=bool)
    allowed[list(available_channels)] = True
    planner, replay = FlyBody(parameters), FlyBody(parameters)
    targets, initial_height = reference_geometry(parameters, available_channels)
    start, goal = planner.data.qpos[planner.qadr].copy(), targets[cue]
    commands, activations, metrics = [], [], []
    for step in range(steps):
        if stop is not None and stop.is_set():
            from .gesture_training import Cancelled
            raise Cancelled()
        if step >= 10 and cue != 'point_both':
            command = commands[-1].copy()
        else:
            if step >= 10 and cue == 'point_both':
                # Continue planning from the independently replayed body's measured
                # state, rather than repeating a transient command indefinitely.
                planner.data.qpos[:] = replay.data.qpos
                planner.data.qvel[:] = replay.data.qvel
                planner.data.act[:] = replay.data.act
                mujoco.mj_forward(planner.model, planner.data)
            pulse = []
            for tick in range(20):
                fraction = min(1., (step * 20 + tick + 1) / 100)
                blend = fraction * fraction * (3 - 2 * fraction)
                desired = start + (goal - start) * blend
                torque = 200 * (desired - planner.data.qpos[planner.qadr]) - .2 * planner.data.qvel[planner.vadr]
                excitation = np.zeros(planner.model.nu)
                excitation[:84] = np.clip(np.stack((np.maximum(torque, 0), np.maximum(-torque, 0)), axis=1)
                    / (MAX_FORCE * parameters.strength_scale * MOMENT_ARM) + .025, 0, 1).ravel()
                excitation[:84] *= allowed
                planner.step_muscles(excitation, dt=PLANNER_DT)
                pulse.append(excitation)
            # Crouch uses the terminal planner command. The other poses use the mean
            # with calibrated antagonist coactivation to stabilize 20 ms open-loop holds.
            command = pulse[-1].copy() if cue == 'fist' else np.mean(pulse, axis=0)
            if cue != 'fist':
                command[:84] = np.clip(command[:84] + .075, 0, 1)
            command[:84] *= allowed
        replay.step_muscles(command, dt=COMMAND_DT)
        commands.append(command)
        activations.append(replay.data.act.copy())
        metrics.append(evaluate_reference(replay, cue, initial_height))
    settled = metrics[4:]  # The first 100 ms includes the movement transition.
    verified = bool(settled and all(m['pose_reached'] for m in settled))
    evidence = {**metrics[-1], 'pose_reached': verified, 'target_version': TARGET_VERSION,
                'command_hz': 50, 'settle_seconds': .1, 'checked_seconds': steps * COMMAND_DT,
                'checked_frames': len(settled), 'source': 'simulated engineering muscle commands',
                'supervised_channels': 84, 'routed_channels': len(available_channels),
                'commanded_channels': list(available_channels), 'hold_starts_seconds': .2}
    return ReferenceRollout(replay, np.array(activations), np.array(commands), metrics, evidence)
