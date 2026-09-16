import numpy as np
import pytest
import torch

from flylab.body import FlyBody
from flylab.calibration import motor_audit, feedback_audit, summarize_motion, run_trial, reflex_score, select_reflex_gain, support_comparison
from flylab.simulation import Simulation
from test_motor_mapping import bridge_for, row
from test_full_connectome import graph_fixture


def test_rate_clamp_restores_state_and_catches_reversed_route():
    body = FlyBody()
    bridge = bridge_for([row('Ti flexor MN'), row('Ti extensor MN', body_id=2)])
    bridge.brain.ids = np.array([1, 2])
    rates = bridge.brain.rates
    before = body.data.qpos.copy()
    result = motor_audit(body, bridge)
    assert result['passed'] == result['mapped'] == 2
    assert bridge.brain.rates is rates
    np.testing.assert_array_equal(body.data.qpos, before)
    assert not body.data.act.any()
    bridge.mapping[0]['transmissions'][0]['torque_sign'] *= -1
    assert motor_audit(body, bridge)['passed'] == 1


def test_feedback_audit_restores_pose_and_covers_both_sides():
    body = FlyBody(); bridge = bridge_for([row('Ti flexor MN')])
    body.data.qvel[body.vadr[5]] = 2.
    pose, velocity = body.data.qpos.copy(), body.data.qvel.copy()
    report = feedback_audit(body, bridge)
    assert len(report['records']) == 6
    assert all(r['flexion_sign_consistent'] for r in report['records'])
    np.testing.assert_array_equal(body.data.qpos, pose)
    np.testing.assert_array_equal(body.data.qvel, velocity)


def samples_for_motion():
    return [{'time_s': i/60, 'foot_normal_uN': [1.]*6,
        'foot_position_mm': [[i*.01, j, 0.] for j in range(6)],
        'feet_supported': True, 'position_mm': [i*.01, 0, 1],
        'tilt_deg': 0., 'other_vertical_uN': 0., 'mean_activation': 0.} for i in range(181)]


def test_sliding_never_counts_as_swing_and_recovery_needs_dwell():
    samples = samples_for_motion()
    result = summarize_motion(samples, 60)
    assert result['candidate_swing_counts'] == [0]*6
    assert result['loaded_foot_travel_mm'] == pytest.approx([1.8]*6)
    assert result['support_recovery_s'] == pytest.approx(0.)
    for s in samples[126:]:
        s['feet_supported'] = False
    samples[-1]['feet_supported'] = True
    assert summarize_motion(samples, 60)['support_recovery_s'] is None


def test_swing_requires_clearance_and_displaced_landing():
    samples = samples_for_motion()
    for i in range(40, 60):
        samples[i]['foot_normal_uN'][0] = 0.
        samples[i]['foot_position_mm'][0][2] = .1
    assert summarize_motion(samples, 60)['candidate_swing_counts'][0] == 1
    for i in range(40, 60):
        samples[i]['foot_position_mm'][0][2] = 0.
    assert summarize_motion(samples, 60)['candidate_swing_counts'][0] == 0


def test_disconnected_control_still_integrates_entire_brain(tmp_path):
    graph_fixture(tmp_path)
    sim = Simulation(); sim.configure('connectome', tmp_path/'full')
    values = sim.full_brain.weights.values().clone()
    report = run_trial(sim, condition='motor_disconnected', seconds=3.)
    assert report['neural_ticks'] == round(3000/sim.full_brain.config.dt_ms)
    assert sim.time == pytest.approx(3.) and not sim.body.data.act.any()
    assert not sim.body.data.xfrc_applied.any()
    torch.testing.assert_close(sim.full_brain.weights.values(), values, rtol=0, atol=0)
    assert 'muscles' in sim.bridge.__dict__  # restored bound method
    assert report['metrics']['walking_validated'] is False


@pytest.mark.parametrize('kwargs', [{'seconds': float('nan')}, {'push_uN': float('inf')},
                                   {'sensory_log_gain': 3}, {'condition': 'bad'}])
def test_rejects_invalid_trial_before_mutation(kwargs):
    with pytest.raises(ValueError):
        run_trial(None, **kwargs)


def test_reflex_fit_uses_training_side_and_penalizes_wrong_responses():
    def candidate(gain, left, right):
        return {'bridge_parameters': [0.,0.,0.,0.,gain,0.,0.,-2.],
                'results': [{'leg': side+'F', 'status': 'measured_model_response',
                             'matches_restoring_hypothesis': value}
                            for side, values in [('L', left), ('R', right)] for value in values]}
    base = candidate(0., [True,True,None,None], [False]*4)
    overdriven = candidate(.5, [True,True,True,False], [True]*4)
    assert select_reflex_gain([overdriven, base]) == 1  # Held-out right side cannot select the fit.
    assert reflex_score(overdriven,'L')['score'] == 1
    assert select_reflex_gain([candidate(.2,[True,True,None,None],[]),base]) == 1
    with pytest.raises(ValueError): select_reflex_gain([])


def test_support_gate_rejects_silence_and_failed_recovery():
    from copy import deepcopy
    baseline = {'seconds': 4., 'push_uN': 3., 'command_hz': 60,
        'metrics': {'mean_muscle_activation': .01, 'feet_supported_fraction': .98,
        'post_push_supported_fraction': .98, 'maximum_tilt_deg': 8.,
        'loaded_foot_travel_mm': [1.]*6, 'maximum_other_support_uN': 0., 'support_recovery_s': .1}}
    candidate = deepcopy(baseline)
    candidate['metrics']['maximum_tilt_deg'] = 5.
    assert support_comparison(candidate, baseline)['passes']
    assert support_comparison(candidate, baseline)['meaningful_improvement']
    candidate['metrics']['mean_muscle_activation'] = 0.
    assert not support_comparison(candidate, baseline)['passes']
    candidate['metrics']['mean_muscle_activation'] = .01
    candidate['metrics']['support_recovery_s'] = None
    assert not support_comparison(candidate, baseline)['passes']
    candidate['push_uN'] = -3.
    with pytest.raises(ValueError): support_comparison(candidate, baseline)


@pytest.mark.parametrize('kwargs', [{'strength_scale': 0}, {'strength_scale': float('nan')},
                                   {'vnc_log_gain': float('inf')}, {'motor_log_gain': 3}])
def test_rejects_invalid_calibration_parameters(kwargs):
    with pytest.raises(ValueError): run_trial(None, **kwargs)


def test_reflex_assay_verifies_target_activation_without_moving_body(tmp_path, monkeypatch):
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.feather as feather
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from probe_reflexes import assay
    graph_fixture(tmp_path)
    path = tmp_path/'full/neurons.feather'
    rows = feather.read_table(path).to_pylist()
    rows[0].update(superclass='vnc_sensory', type='SNpp50',
                   entryNerve='ProLN', **{'class': 'mechanosensory_proprioceptive'})
    feather.write_feather(pa.Table.from_pylist(rows), path)
    sim = Simulation(); sim.configure('connectome', tmp_path/'full')
    report = assay(sim)
    tested = [r for r in report['results'] if r['status'] != 'unmapped']
    assert len(tested) == 1 and tested[0]['body_ids'] == [10]
    assert tested[0]['peak_target_mean_rate_increase_hz'] > 0
    assert sim.body.data.time == 0 and not sim.body.data.act.any()
    assert report['pulse_current_mV'] == 14.
    with pytest.raises(ValueError):
        assay(None, pulse_mV=float('nan'))
    with pytest.raises(ValueError):
        assay(None, vnc_log_gain=float('nan'))
