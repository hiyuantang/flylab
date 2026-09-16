"""Offline causal checks for the measured brain/body interface.

Direct rate clamps are diagnostic interventions, not locomotion controllers.
Trial ablations never remove neurons, edges, or neural integration steps.
"""
from dataclasses import asdict, replace
import hashlib
import time

import mujoco
import numpy as np
import torch

from .body import LEGS
from .leg_feedback import sample_legs
from .senses import SensorSuite


def reflex_score(report, side):
    """Exploratory loss: wrong-direction responses cost twice a gain.

    Counts are over available groups on one side, never neuron-weighted. This is
    a qualitative model target, not a fit to measured biological torque traces.
    """
    if side not in {'L', 'R'}:
        raise ValueError('Expected L or R')
    records = [r for r in report['results'] if r['leg'].startswith(side) and r['status'] != 'unmapped']
    restoring = sum(r.get('matches_restoring_hypothesis') is True for r in records)
    opposing = sum(r.get('matches_restoring_hypothesis') is False for r in records)
    return {'available': len(records), 'restoring': restoring, 'opposing': opposing,
            'silent': len(records)-restoring-opposing, 'score': restoring-2*opposing}


def select_reflex_gain(reports):
    """Select using left legs only; ties retain the smallest parameter change."""
    if not reports:
        raise ValueError('No calibration candidates')
    return max(range(len(reports)), key=lambda i: (reflex_score(reports[i], 'L')['score'],
                                                 -abs(reports[i]['bridge_parameters'][4])))


def support_comparison(candidate, baseline):
    """Explicit engineering acceptance tolerances, not statistical confidence.

    Compare identical durations/perturbations. Silence and failed recovery must
    never win because of a lower movement/tilt score.
    """
    for key in ('seconds', 'push_uN', 'command_hz'):
        if candidate[key] != baseline[key]:
            raise ValueError('Support comparison requires matched conditions')
    a, b = candidate['metrics'], baseline['metrics']
    checks = {
        'muscles_active': a['mean_muscle_activation'] > 1e-8,
        'support_retained': a['feet_supported_fraction'] >= b['feet_supported_fraction']-.01,
        'post_push_support_retained': a['post_push_supported_fraction'] >= b['post_push_supported_fraction']-.02,
        'tilt_not_worse': a['maximum_tilt_deg'] <= b['maximum_tilt_deg']+.5,
        'loaded_foot_motion_not_worse': sum(a['loaded_foot_travel_mm']) <= 1.05*sum(b['loaded_foot_travel_mm'])+1e-8,
        'body_contact_not_worse': a['maximum_other_support_uN'] <= b['maximum_other_support_uN']+.01,
        'recovery_retained': b['support_recovery_s'] is None or a['support_recovery_s'] is not None,
    }
    improved = (a['feet_supported_fraction'] > b['feet_supported_fraction']+.01 or
                a['maximum_tilt_deg'] < .9*b['maximum_tilt_deg'] or
                sum(a['loaded_foot_travel_mm']) < .9*sum(b['loaded_foot_travel_mm']))
    return {'checks': checks, 'passes': all(checks.values()), 'meaningful_improvement': improved}


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def motor_audit(body, bridge, rate_hz=100.):
    """Clamp one annotated MN rate, inspect actual MuJoCo actuator torque.

    Fixed-pose, steady activation assay: it measures transmission consistency,
    not a physiological firing-to-force curve or free-body movement.
    """
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError('Rate must be finite and positive')
    brain = bridge.brain
    saved_rates, saved_act = brain.rates, body.data.act.copy()
    index = {int(value): i for i, value in enumerate(brain.ids)}
    brain.rates = torch.zeros(len(brain.ids), dtype=torch.float64)
    records = []
    try:
        body.data.act[:] = 0
        mujoco.mj_forward(body.model, body.data)
        baseline = body.data.qfrc_actuator.copy()
        for record in bridge.mapping:
            i = index[record['body_id']]
            brain.rates[i] = rate_hz
            excitation = bridge.muscles()
            brain.rates[i] = 0
            body.data.act[:] = excitation
            mujoco.mj_forward(body.model, body.data)
            torque = body.data.qfrc_actuator - baseline
            checks, expected_dofs = [], set()
            for transmission in record['transmissions']:
                joint = body.model.joint(transmission['joint'])
                dof = int(joint.dofadr[0]); expected_dofs.add(dof)
                measured = float(torque[dof])
                checks.append({**transmission, 'generalized_force': measured,
                    'unit': 'uN' if body.model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_SLIDE else 'uN mm',
                    'direction_matches_route': measured * transmission['torque_sign'] > 1e-10})
            unexpected = [int(i) for i in np.flatnonzero(np.abs(torque) > 1e-9) if i not in expected_dofs]
            records.append({'body_id': record['body_id'], 'type': record['type'],
                'leg_or_side': record['leg'], 'checks': checks,
                'excited_channels': np.flatnonzero(excitation).tolist(),
                'maximum_excitation': float(excitation.max()), 'unexpected_force_dofs': unexpected,
                'passed': all(c['direction_matches_route'] for c in checks) and not unexpected})
    finally:
        brain.rates = saved_rates
        body.data.act[:] = saved_act
        mujoco.mj_forward(body.model, body.data)
    return {'scope': 'Individual MN rate clamp -> existing pooled muscle readout -> fixed-pose actuator force. This checks code consistency, not anatomical validity.',
        'rate_hz': rate_hz, 'motor_log_gain': float(bridge.parameters[7]),
        'mapped': len(records), 'passed': sum(r['passed'] for r in records),
        'unmapped': bridge.unmapped, 'records': records}


def feedback_audit(body, bridge):
    """Perturb each actual knee coordinate; report sign and mapped receptor IDs."""
    qpos, qvel = body.data.qpos.copy(), body.data.qvel.copy()
    records = []
    try:
        body.data.qvel[:] = 0
        mujoco.mj_forward(body.model, body.data)
        base = sample_legs(body)
        for leg_index, leg in enumerate(LEGS):
            joint = body.model.joint(leg.lower() + '_tibia_pitch')
            body.data.qpos[:] = qpos
            body.data.qpos[joint.qposadr[0]] += .05
            body.data.qvel[:] = 0
            body.data.qvel[joint.dofadr[0]] = 1.
            mujoco.mj_forward(body.model, body.data)
            frame = sample_legs(body)
            change = float(frame['opening_rad'][leg_index] - base['opening_rad'][leg_index])
            drive = torch.zeros(len(bridge.brain.ids))
            bridge.leg_feedback.apply(drive, frame, 1.)
            groups = {channel: [int(bridge.brain.ids[i]) for i in indices]
                for (j, channel), indices in bridge.leg_feedback.groups.items() if j == leg_index}
            records.append({'leg': leg, 'joint_delta_rad': .05, 'opening_delta_rad': change,
                'opening_velocity_rad_s': frame['velocity_rad_s'][leg_index],
                'signals': {key: value[leg_index] for key, value in frame['signals'].items()},
                'receptor_body_ids': groups,
                'flexion_sign_consistent': change < 0 and frame['velocity_rad_s'][leg_index] < 0})
    finally:
        body.data.qpos[:], body.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(body.model, body.data)
    return {'scope': 'Knee kinematics and annotated receptor coverage. Tuning curves and reflex efficacy remain hypotheses.',
            'routing': bridge.leg_feedback.summary(), 'records': records}


CONDITIONS = ('intact', 'no_proprioception', 'no_touch', 'no_feedback', 'motor_disconnected')


def summarize_motion(samples, hz, push_end=2.1):
    """Contact-aware diagnostics; displacement alone never qualifies as walking."""
    settled = [s for s in samples if s['time_s'] >= .5]
    if not settled:
        raise ValueError('Need samples after 0.5 seconds')
    contacts = np.array([s['foot_normal_uN'] for s in samples]) > .02
    feet = np.array([s['foot_position_mm'] for s in samples])
    delta = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
    loaded = contacts[1:] & contacts[:-1]
    slips = np.sum(np.where(loaded, delta, 0.), axis=0)
    swings = np.zeros(6, dtype=int)
    for leg in range(6):
        start = None
        for i in range(1, len(samples)):
            if contacts[i-1, leg] and not contacts[i, leg]:
                start = i-1
            elif start is not None and contacts[i, leg]:
                path = feet[start:i+1, leg]
                if i-start >= 3 and np.max(path[:, 2])-max(path[0, 2], path[-1, 2]) >= .03 and np.linalg.norm(path[-1, :2]-path[0, :2]) >= .1:
                    swings[leg] += 1
                start = None
    recovery = None
    window = max(1, round(.25*hz))
    for i, s in enumerate(samples):
        if s['time_s'] >= push_end and i+window <= len(samples) and all(x['feet_supported'] for x in samples[i:i+window]):
            recovery = max(0., s['time_s'] - push_end)
            break
    post = [s for s in settled if s['time_s'] >= push_end]
    return {'feet_supported_fraction': float(np.mean([s['feet_supported'] for s in settled])),
        'post_push_supported_fraction': float(np.mean([s['feet_supported'] for s in post])) if post else None,
        'support_recovery_s': recovery,
        'maximum_tilt_deg': max(s['tilt_deg'] for s in settled),
        'maximum_other_support_uN': max(s['other_vertical_uN'] for s in settled),
        'net_xy_mm': float(np.linalg.norm(np.array(samples[-1]['position_mm'][:2])-samples[0]['position_mm'][:2])),
        'loaded_foot_travel_mm': slips.tolist(), 'candidate_swing_counts': swings.tolist(),
        'mean_muscle_activation': float(np.mean([s['mean_activation'] for s in settled])),
        'walking_validated': False,
        'interpretation': 'Swing candidates require unload, clearance and displaced landing; these thresholds are engineering diagnostics, not biological gait validation. Loaded-foot travel includes contact deformation and sliding.'}


def run_trial(sim, *, condition='intact', seconds=4., push_uN=3., sensory_log_gain=0.,
              descending_drive=0., progress=None, vnc_log_gain=0., motor_log_gain=-2., strength_scale=1.):
    """Matched reset, fixed clock, full graph in every condition (even disconnected).

    Only motor_disconnected replaces the delivered muscle command with zero;
    its brain still computes every neural step. No changes reach the live app.
    """
    if condition not in CONDITIONS:
        raise ValueError('Unknown experimental condition')
    if not all(np.isfinite(x) for x in (seconds, push_uN, sensory_log_gain, descending_drive, vnc_log_gain, motor_log_gain, strength_scale)) or not 3 <= seconds <= 30 or abs(push_uN) > 10 or not -2 <= sensory_log_gain <= 2 or not 0 <= descending_drive <= 3 or not -2 <= vnc_log_gain <= 2 or not -2 <= motor_log_gain <= 2 or not .05 <= strength_scale <= 4:
        raise ValueError('Invalid trial parameters')
    hz = sim.command_hz
    cycles = round(seconds*hz)
    if not np.isclose(cycles/hz, seconds):
        raise ValueError('Duration must align with the command clock')
    sim.prepare_standing_trial()
    if strength_scale != 1.:
        from .body import FlyBody
        sim.body = FlyBody(replace(sim.body.parameters, strength_scale=strength_scale), 'lab')
        sim.reset()
    sim.sensors = SensorSuite(replace(sim.sensors.settings,
        proprioception_enabled=condition not in {'no_proprioception', 'no_feedback'},
        touch_enabled=condition not in {'no_touch', 'no_feedback'}))
    parameters = sim.bridge.parameters.copy(); parameters[6] = sensory_log_gain
    parameters[4], parameters[7] = vnc_log_gain, motor_log_gain
    sim.bridge.set_parameters(parameters)
    if descending_drive and not len(sim.bridge.locomotion.groups['DNg100']):
        raise ValueError('DNg100 neurons unavailable')
    body, brain = sim.body, sim.full_brain
    initial = {'qpos_sha256': digest_array(body.data.qpos),
               'voltage_sha256': digest_array(brain.voltage.cpu().numpy()), 'tick_index': brain.tick_index}
    original_muscles = sim.bridge.muscles
    if condition == 'motor_disconnected':
        sim.bridge.muscles = lambda: np.zeros(body.model.nu, dtype=np.float32)
    samples = []
    def sample():
        result = body.support_snapshot()
        result.update(time_s=float(body.data.time), position_mm=body.data.qpos[:3].tolist(),
            foot_position_mm=body.data.site_xpos[body.foot_ids].tolist(),
            tilt_deg=float(np.rad2deg(np.arccos(np.clip(body.data.xmat[body.body_ids['c_thorax']].reshape(3,3)[2,2], -1, 1)))),
            mean_activation=float(body.data.act.mean()))
        return result
    samples.append(sample()); started = time.perf_counter()
    try:
        for i in range(cycles):
            t = i/hz
            body.data.xfrc_applied[body.body_ids['c_thorax'], 1] = push_uN if 2 <= t < 2.1 else 0.
            if descending_drive and t >= .5 and (not sim.bridge.locomotion.amplitude or t >= sim.bridge.locomotion.expires):
                sim.bridge.locomotion.submit('DNg100', descending_drive, 5., sim.time)
            sim.advance()
            samples.append(sample())
            if progress and (i+1) % hz == 0:
                progress(f'{condition}, gain={sensory_log_gain:g}, push={push_uN:g}, DN={descending_drive:g}: {(i+1)/hz:g}/{seconds:g} simulated seconds')
    finally:
        sim.bridge.muscles = original_muscles
        body.data.xfrc_applied[:] = 0
    ticks = brain.tick_index - initial['tick_index']
    expected = round(seconds*1000/brain.config.dt_ms)
    if ticks != expected:
        raise RuntimeError(f'Neural clock lost steps: {ticks} != {expected}')
    return {'condition': condition, 'seconds': seconds, 'push_uN': push_uN,
        'push_interval_s': [2., 2.1], 'sensory_log_gain': sensory_log_gain,
        'descending_drive': descending_drive, 'initial_state': initial,
        'vnc_log_gain': vnc_log_gain, 'motor_log_gain': motor_log_gain,
        'wall_seconds': time.perf_counter()-started, 'neural_ticks': ticks,
        'neurons': len(brain.ids), 'edges': brain.manifest['edges'],
        'graph_sha256': brain.manifest['graph_sha256'], 'execution': brain.execution_summary(),
        'physiology': asdict(brain.config), 'body_parameters': asdict(body.parameters),
        'sensory_settings': asdict(sim.sensors.settings), 'bridge_parameters': sim.bridge.parameters.tolist(),
        'command_hz': hz, 'coupling': sim.coupling_mode,
        'metrics': summarize_motion(samples, hz), 'samples': samples}
