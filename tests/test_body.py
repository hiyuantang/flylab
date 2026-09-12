import numpy as np
import mujoco
from flylab.body import FlyBody, SEGMENTS
from flylab.arena import odor_sensors


def test_articulated_hierarchy_preserves_every_attachment_during_motion():
    body = FlyBody()
    assert body.model.nbody - 1 == 69
    assert body.model.njnt - 1 == 70
    assert body.model.nu == 84
    for _ in range(20):
        body.step(np.full(12, .4))
    for name, config in SEGMENTS.items():
        if config['parent'] is None:
            continue
        child = body.body_ids[name]
        parent = body.body_ids[config['parent']]
        assert body.model.body_parentid[child] == parent
        expected = body.data.xpos[parent] + body.data.xmat[parent].reshape(3, 3) @ body.model.body_pos[child]
        np.testing.assert_allclose(body.data.xpos[child], expected, atol=1e-10)
    assert .8 < body.snapshot()['anatomy']['mass_mg'] < 1.3


def test_muscle_forces_produce_upright_forward_locomotion_and_decay():
    body = FlyBody()
    start = body.data.qpos[:3].copy()
    self_contact_force = 0.
    contact_force = np.zeros(6)
    for _ in range(100):
        body.step(np.full(12, .4))
        assert body.snapshot()['upright'] > .85
        for index, contact in enumerate(body.data.contact):
            if contact.geom1 and contact.geom2:
                # Allow small soft-contact deformation, never visible pass-through.
                assert contact.dist > -.015  # mm, 15 µm
                mujoco.mj_contactForce(body.model, body.data, index, contact_force)
                self_contact_force = max(self_contact_force, contact_force[0])
    assert body.data.qpos[0] - start[0] > 5
    assert np.max(body.data.actuator_force) <= 0  # Hill muscles pull, never push.
    assert np.abs(body.data.actuator_force).sum() > 1
    assert body.foot_feedback().sum() > 0
    assert self_contact_force > 0
    for _ in range(20):
        body.step(np.zeros(12))
    assert body.data.ctrl.max() == 0
    assert body.data.act.max() < 1e-6
    assert np.isfinite(body.data.qpos).all()


def test_reset_is_free_of_self_intersections_and_wings_react_to_contact():
    body = FlyBody()
    m, d = body.model, body.data
    assert all(c.dist >= .001 for c in d.contact if c.geom1 and c.geom2)
    assert d.time == 0 and np.all(d.qvel == 0)
    # Press both wings inward within their joint limits; their surfaces must
    # generate repulsive contact forces, then separate under physical integration.
    for side in ['l', 'r']:
        d.qpos[m.joint(f'{side}_wing_spread').qposadr] = -.05
    mujoco.mj_forward(m, d)
    force = np.zeros(6)
    wing_contact = False
    for index, contact in enumerate(d.contact):
        names = [m.geom(int(g)).name for g in contact.geom]
        if all('wing_surface' in name for name in names) and contact.dist < 0:
            mujoco.mj_contactForce(m, d, index, force)
            wing_contact |= force[0] > 0
    assert wing_contact
    body.step_muscles(np.zeros(84), dt=.04)
    assert all(c.dist > -.015 for c in d.contact if c.geom1 and c.geom2)


def test_odor_is_sampled_at_moving_antennae_and_falls_with_distance():
    body = FlyBody()
    close = odor_sensors(body, [1, .2, 0])
    far = odor_sensors(body, [30, .2, 0])
    assert np.all(close > far)
    assert close[0] != close[1]
    initial = odor_sensors(body, [8, 0, 0])
    body.data.qpos[0] += 4
    import mujoco
    mujoco.mj_forward(body.model, body.data)
    assert np.all(odor_sensors(body, [8, 0, 0]) > initial)
