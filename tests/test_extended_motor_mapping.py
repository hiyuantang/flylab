from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from flylab.body import BodyParameters, FlyBody
from flylab.extended_mechanics import ADDITIONAL_MUSCLES, MUSCLES, PROFILE, MODEL, CHANNEL_COUNT
from flylab.motor_mapping import resolve_motor, PERIPHERAL_PROFILE
from flylab.neuromuscular import NeuromuscularBridge
from flylab.pretarsus import migrate_body


def row(name, subclass, side='L', body_id=1):
    return dict(bodyId=body_id, type=name, superclass='cb_motor' if name.startswith('CvN') else 'vnc_motor',
                subclass=subclass, somaSide=side, instance=f'{name}_{side}')


def bridge(rows, profile=PROFILE):
    brain = SimpleNamespace(neurons=rows, ids=np.arange(len(rows)), voltage=torch.zeros(len(rows)),
                            rates=torch.zeros(len(rows)))
    return NeuromuscularBridge(brain, profile)


def test_additional_outputs_have_specific_targets_provenance_and_force_paths():
    rows = [row(m.neuron_type, m.subclass, m.side, i+1) for i, m in enumerate(ADDITIONAL_MUSCLES)]
    b = bridge(rows)
    body = FlyBody(BodyParameters(appendage_model=MODEL))
    assert body.model.nu == len(b.channels) == CHANNEL_COUNT == 196
    for index, muscle in enumerate(ADDITIONAL_MUSCLES):
        b.brain.rates.zero_()
        b.brain.rates[index] = 100
        action = b.muscles()
        channel = body.model.actuator(muscle.name).id
        assert np.flatnonzero(action).tolist() == [channel]
        body.data.act[:] = 0
        mujoco.mj_forward(body.model, body.data)
        before = body.data.qfrc_actuator.copy()
        body.data.act[channel] = 1
        mujoco.mj_forward(body.model, body.data)
        for joint, coefficient in muscle.joints:
            dof = body.model.joint(joint).dofadr[0]
            assert (body.data.qfrc_actuator[dof] - before[dof]) * coefficient < 0
        record = b.mapping[index]
        assert record['sources'] and record['confidence']['mechanics']['level'] == 'approximate'
        assert resolve_motor(rows[index], PERIPHERAL_PROFILE)['status'] == 'unmapped'


def test_vl1_pooling_does_not_duplicate_the_muscle_or_amplify_multiple_neurons():
    b = bridge([row(f'CvN{i}', 'nm', body_id=i) for i in range(4, 8)])
    b.brain.rates[:] = 100
    action = b.muscles()
    assert len(np.flatnonzero(action)) == 1
    assert action.sum() == pytest.approx(1)
    assert all(m['muscle_target'] == 'VL1' for m in b.mapping)
    b.brain.rates[:3] = 0
    assert b.muscles().sum() == pytest.approx(.25)


def test_crossed_adductor_and_uncertain_target_labels_are_not_hidden():
    for side, target, sign in [('L', 'R', -1), ('R', 'L', 1)]:
        record = bridge([row('FNM2', 'nm', side)]).mapping[0]
        assert record['target_side'] == target
        assert record['transmissions'][0]['torque_sign'] == sign
        assert record['confidence']['identity']['level'] == 'tentative'
    for name in ['MNhm42', 'MNhm43']:
        record = resolve_motor(row(name, 'hm'), PROFILE)
        assert record['muscle_target'] == 'hb1 or hb2'
        assert record['confidence']['identity']['level'] == 'family_only'
    record = resolve_motor(row('MNwm35', 'wm'), PROFILE)
    assert record['confidence']['identity']['level'] == 'tentative'
    assert record['reference_matches'][0]['match_certainty(1-5)'] == '1'
    for name, region in [('MNwm36', 'wm'), ('MNad21', 'ad'), ('MNml82', 'ml'), ('Fe reductor MN', 'fl')]:
        assert resolve_motor(row(name, region), PROFILE)['status'] == 'unmapped'
    conflict = row('CvN4', 'nm'); conflict['instance'] = 'CvN4_R'
    assert resolve_motor(conflict, PROFILE)['reason'] == 'annotation_conflict'


def test_new_neck_pitch_physically_lowers_the_front_of_the_head():
    body = FlyBody(BodyParameters(appendage_model=MODEL))
    m, d = body.model, body.data
    def forward_point():
        return d.xpos[m.body('c_head').id] + d.xmat[m.body('c_head').id].reshape(3, 3) @ np.array([.2, 0, 0])
    before = forward_point()
    d.qpos[m.joint('c_head_pitch').qposadr[0]] += .01
    mujoco.mj_forward(m, d)
    assert forward_point()[2] < before[2]


def test_v4_to_v5_named_state_migration_and_checkpoint_continuation(tmp_path):
    from test_full_connectome import graph_fixture
    from flylab.full_connectome import Physiology
    from flylab.simulation import Simulation
    from flylab.live_state import save_live, load_live, NEURAL_TENSORS
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path/'full', Physiology.paper(),
                  BodyParameters(appendage_model='peripheral-v1'), mapping_profile=PERIPHERAL_PROFILE)
    sim.advance()
    old = sim.body
    old.data.act[:] = np.linspace(0, .01, old.model.nu)
    neurons = {name: getattr(sim.full_brain, name).clone() for name in NEURAL_TENSORS}
    new = migrate_body(old, FlyBody(BodyParameters(appendage_model=MODEL)))
    assert old.model.nu == 186 and new.model.nu == 196
    for i in range(old.model.nu):
        assert old.model.actuator(i).name == new.model.actuator(i).name
        assert old.data.act[i] == new.data.act[i]
    assert np.count_nonzero(new.data.act[186:]) == 0
    for i in range(old.model.njnt):
        joint = old.model.joint(i)
        n = 7 if joint.type[0] == mujoco.mjtJoint.mjJNT_FREE else 1
        target = new.model.joint(joint.name)
        np.testing.assert_array_equal(old.data.qpos[joint.qposadr[0]:joint.qposadr[0]+n],
                                      new.data.qpos[target.qposadr[0]:target.qposadr[0]+n])
    sim.body = new
    sim.bridge = NeuromuscularBridge(sim.full_brain, PROFILE)
    for name, value in neurons.items():
        torch.testing.assert_close(value, getattr(sim.full_brain, name), rtol=0, atol=0)
    path = tmp_path/'extended.pt'
    save_live(sim, path)
    restored = load_live(path, tmp_path/'full')
    sim.advance(); restored.advance()
    for name in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, name), getattr(restored.full_brain, name), rtol=0, atol=0)
    np.testing.assert_array_equal(sim.body.data.qpos, restored.body.data.qpos)


def test_installed_annotations_add_exactly_the_reviewed_neurons():
    from pathlib import Path
    import pyarrow.feather as feather
    path = Path('data/full/neurons.feather')
    if not path.exists(): pytest.skip('Measured annotations not installed')
    rows = [r for r in feather.read_table(path).to_pylist() if r['superclass'] in {'cb_motor', 'vnc_motor'}]
    old = {r['bodyId'] for r in rows if resolve_motor(r, PERIPHERAL_PROFILE)['status'] == 'mapped'}
    new = {r['bodyId'] for r in rows if resolve_motor(r, PROFILE)['status'] == 'mapped'}
    assert len(old) == 438 and len(new) == 454 and old <= new
    assert {r['type'] for r in rows if r['bodyId'] in new-old} == {
        'CvN4', 'CvN5', 'CvN6', 'CvN7', 'FNM2', 'MNwm35', 'MNhm42', 'MNhm43'}
