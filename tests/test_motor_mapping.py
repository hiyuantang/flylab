from types import SimpleNamespace
import numpy as np
import pytest
import torch
import mujoco
from flylab.motor_mapping import resolve_motor, TARGETS, MAPPING_PROFILE, LEGACY_PROFILE
from flylab.neuromuscular import NeuromuscularBridge
from flylab.body import FlyBody, LEGS


def row(name, leg='LF', body_id=1, **extra):
    return {'bodyId': body_id, 'type': name, 'superclass': 'vnc_motor',
            'class': None, 'subclass': {'F': 'fl', 'M': 'ml', 'H': 'hl'}[leg[1]],
            'somaSide': leg[0], 'instance': f'{name}_{leg[0]}', **extra}


def bridge_for(rows, profile=MAPPING_PROFILE):
    brain = SimpleNamespace(neurons=rows, ids=np.arange(len(rows)),
                            voltage=torch.zeros(len(rows), dtype=torch.float64),
                            rates=torch.full((len(rows),), 100., dtype=torch.float64))
    return NeuromuscularBridge(brain, profile)


def test_inventory_includes_head_and_never_guesses_missing_anatomy():
    rows = [row('MN9', superclass='cb_motor', subclass='pm', body_id=1),
            row('Fe reductor MN', body_id=2), row('ltm1-tibia MN', body_id=3),
            row('MNhl87', 'LH', body_id=4), row('Ti extensor MN', body_id=5)]
    bridge = bridge_for(rows)
    s = bridge.summary()
    assert s['total_motor_neurons'] == 5 and s['mapped_motor_neurons'] == 1
    assert set(s['unmapped_reasons']) == {'body_actuator_missing', 'muscle_action_unknown', 'pretarsal_tendon_missing', 'muscle_identity_unresolved'}
    assert np.count_nonzero(bridge.muscles()) == 1
    for changes in [{'somaSide': None}, {'instance': 'wrong_R'}, {'exitNerve': 'MetaLN'}]:
        assert resolve_motor(row('Ti flexor MN', **changes))['status'] == 'unmapped'
    # Nerves can exit outside the leg neuromere; do not reject a known exception.
    assert resolve_motor(row('Tergotr. MN', 'LH', exitNerve='AbN1'))['status'] == 'mapped'


@pytest.mark.parametrize('leg', LEGS)
def test_new_connections_actuate_only_the_annotated_leg_and_both_remotor_axes(leg):
    for name in TARGETS:
        bridge = bridge_for([row(name, leg)])
        action = bridge.muscles()
        active = np.flatnonzero(action)
        assert len(active) == (2 if name == 'Pleural remotor/abductor MN' else 1)
        assert set(active // 14) == {LEGS.index(leg)}
        assert action.sum() == pytest.approx(1.)
    legacy = bridge_for([row('Ti extensor MN', leg)], LEGACY_PROFILE)
    current = bridge_for([row('Ti extensor MN', leg)])
    assert np.argmax(legacy.muscles()) == LEGS.index(leg)*14+10
    assert np.argmax(current.muscles()) == LEGS.index(leg)*14+11


@pytest.mark.parametrize('leg', LEGS)
def test_functional_directions_against_rig_geometry_and_actual_muscle_torque(leg):
    body = FlyBody()
    m, d = body.model, body.data
    prefix = leg.lower()+'_'
    def opening(parent, child, end):
        p, c, e = [d.xpos[m.body(prefix+x).id].copy() for x in (parent, child, end)]
        u, v = p-c, e-c
        return np.arccos(np.clip(u@v / (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))
    checks = [
        ('Ti extensor MN', ('trochanterfemur', 'tibia', 'tarsus1'), 1),
        ('Ti flexor MN', ('trochanterfemur', 'tibia', 'tarsus1'), -1),
        ('Tr extensor MN', ('coxa', 'trochanterfemur', 'tibia'), 1),
        ('Sternotrochanter MN', ('coxa', 'trochanterfemur', 'tibia'), 1),
        ('Tergotr. MN', ('coxa', 'trochanterfemur', 'tibia'), 1),
        ('Ta depressor MN', ('tibia', 'tarsus1', 'tarsus5'), 1),
        ('Ta levator MN', ('tibia', 'tarsus1', 'tarsus5'), -1)]
    for name, points, expected in checks:
        bridge = bridge_for([row(name, leg)])
        t = bridge.mapping[0]['transmissions'][0]
        joint = m.joint(t['joint'])
        before = opening(*points)
        d.qpos[joint.qposadr[0]] += t['torque_sign']*1e-4
        mujoco.mj_forward(m, d)
        assert (opening(*points)-before)*expected > 0
        d.qpos[joint.qposadr[0]] -= t['torque_sign']*1e-4
        d.act[:] = 0
        mujoco.mj_forward(m, d)
        baseline = d.qfrc_actuator[joint.dofadr[0]]
        d.act[t['actuator']] = 1
        mujoco.mj_forward(m, d)
        assert (d.qfrc_actuator[joint.dofadr[0]]-baseline)*t['torque_sign'] > 0
        d.act[:] = 0
    # Coxa directions are defined in the thorax frame, not screen coordinates.
    rot = d.xmat[m.body('c_thorax').id].reshape(3, 3).copy()
    for name, dim, sign in [('Tergopleural/Pleural promotor MN', 0, 1),
                            ('Sternal posterior rotator MN', 0, -1),
                            ('Sternal adductor MN', 1, -1 if leg[0]=='L' else 1)]:
        t = bridge_for([row(name, leg)]).mapping[0]['transmissions'][0]
        j = m.joint(t['joint']).qposadr[0]
        end = m.body(prefix+'trochanterfemur').id
        before = d.xpos[end].copy()
        d.qpos[j] += t['torque_sign']*1e-4
        mujoco.mj_forward(m, d)
        assert (rot.T@(d.xpos[end]-before))[dim]*sign > 0
        d.qpos[j] -= t['torque_sign']*1e-4
        mujoco.mj_forward(m, d)
