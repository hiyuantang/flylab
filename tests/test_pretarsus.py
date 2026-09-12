import numpy as np
import mujoco
import pytest
from flylab.body import FlyBody, BodyParameters, LEGS
from flylab.pretarsus import migrate_body
from flylab.motor_mapping import PRETARSAL_PROFILE, validate_body_profile
from flylab.env import FlyEnv
from test_motor_mapping import bridge_for, row

PARAMETERS = BodyParameters(appendage_model='pretarsal-v1')


def test_named_long_tendon_neurons_reach_only_their_leg_and_unknowns_stay_unmapped():
    rows = [row('ltm MN', leg, i) for i, leg in enumerate(LEGS)]
    rows += [row('ltm-unknown MN', body_id=9), row('MNhl87', 'LH', body_id=10)]
    bridge = bridge_for(rows, PRETARSAL_PROFILE)
    assert bridge.muscles().shape == (90,)
    np.testing.assert_array_equal(bridge.muscles()[:84], 0)
    np.testing.assert_array_equal(bridge.muscles()[84:], 1)
    assert bridge.summary()['mapped_motor_neurons'] == 6
    assert len(bridge.unmapped) == 2
    with pytest.raises(ValueError, match='do not match'):
        validate_body_profile(PRETARSAL_PROFILE, BodyParameters())


def test_tendon_shortens_with_claw_flexion_and_muscle_produces_pulling_torque():
    body = FlyBody(PARAMETERS)
    m, d = body.model, body.data
    for i, leg in enumerate(LEGS):
        j = m.joint(leg.lower()+'_pretarsus_flexion')
        t = m.tendon(leg.lower()+'_pretarsus_tendon')
        before = d.ten_length[t.id]
        d.qpos[j.qposadr] += .01
        mujoco.mj_forward(m, d)
        assert d.ten_length[t.id] < before
        d.act[:] = 0
        mujoco.mj_forward(m, d)
        baseline = d.qfrc_actuator[j.dofadr].copy()
        d.act[84+i] = 1
        mujoco.mj_forward(m, d)
        assert d.actuator_force[84+i] < 0
        assert (d.qfrc_actuator[j.dofadr] - baseline)[0] > 0
        d.act[:] = 0
    # Real integration changes the joints; a zero command permits elastic return.
    # Lift the fly clear of contact to isolate the distal actuator response.
    body.reset()
    d.qpos[2] += 10
    body.step_muscles(np.r_[np.zeros(84), np.ones(6)], .03)
    joints = [m.joint(leg.lower()+'_pretarsus_flexion').qposadr[0] for leg in LEGS]
    flexed = d.qpos[joints].copy()
    assert np.all(flexed > .1)
    assert np.all(flexed < 1.11)
    body.step_muscles(np.zeros(90), .05)
    assert np.all(d.qpos[joints] < flexed)
    assert np.isfinite(d.qpos).all()
    assert len(body.snapshot()['pretarsi']) == 6


def test_migration_preserves_existing_named_dofs_muscles_clock_and_refuses_removal():
    old = FlyBody()
    old.step_muscles(np.linspace(0, .03, 84), .001)
    new = migrate_body(old, FlyBody(PARAMETERS))
    assert new.data.time == old.data.time
    for i in range(old.model.njnt):
        j = new.model.joint(old.model.joint(i).name)
        nq, nv = (7, 6) if i == 0 else (1, 1)
        a, b = old.model.jnt_qposadr[i], j.qposadr[0]
        np.testing.assert_array_equal(new.data.qpos[b:b+nq], old.data.qpos[a:a+nq])
        a, b = old.model.jnt_dofadr[i], j.dofadr[0]
        np.testing.assert_array_equal(new.data.qvel[b:b+nv], old.data.qvel[a:a+nv])
        np.testing.assert_array_equal(new.data.qacc_warmstart[b:b+nv], old.data.qacc_warmstart[a:a+nv])
    np.testing.assert_array_equal(new.data.act[:84], old.data.act)
    np.testing.assert_array_equal(new.data.ctrl[:84], old.data.ctrl)
    np.testing.assert_array_equal(new.data.act[84:], 0)
    with pytest.raises(ValueError, match='Cannot remove'):
        migrate_body(new, FlyBody())


def test_rl_environment_adapts_action_and_observation_to_installed_mechanics():
    env = FlyEnv(action_mode='muscle', body_parameters=PARAMETERS)
    obs, _ = env.reset(seed=9)
    assert env.action_space.shape == (90,)
    assert env.observation_space.contains(obs)
    obs, _, _, _, _ = env.step(np.zeros(90, dtype=np.float32))
    assert env.observation_space.contains(obs)
    with pytest.raises(ValueError, match='90 finite'):
        env.body.step_muscles(np.zeros(84))


def test_upgraded_checkpoint_continues_exactly_and_baseline_model_is_unchanged(tmp_path):
    from test_full_connectome import graph_fixture
    from flylab.simulation import Simulation
    from flylab.full_connectome import Physiology
    from flylab.live_state import save_live, load_live, NEURAL_TENSORS
    from flylab.neuromuscular import NeuromuscularBridge
    import torch
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', Physiology.paper())
    sim.advance()
    neural = {name: getattr(sim.full_brain, name).clone() for name in NEURAL_TENSORS}
    sim.body = migrate_body(sim.body, FlyBody(PARAMETERS))
    sim.bridge = NeuromuscularBridge(sim.full_brain, PRETARSAL_PROFILE)
    for name, value in neural.items():
        torch.testing.assert_close(getattr(sim.full_brain, name), value, rtol=0, atol=0)
    path = tmp_path / 'pretarsal.pt'
    save_live(sim, path)
    restored = load_live(path, tmp_path / 'full')
    assert restored.body.model.nu == 90
    assert restored.bridge.mapping_profile == PRETARSAL_PROFILE
    sim.advance()
    restored.advance()
    for name in NEURAL_TENSORS:
        torch.testing.assert_close(getattr(sim.full_brain, name), getattr(restored.full_brain, name), rtol=0, atol=0)
    for name in ('qpos', 'qvel', 'act', 'qacc_warmstart'):
        np.testing.assert_array_equal(getattr(sim.body.data, name), getattr(restored.body.data, name))
