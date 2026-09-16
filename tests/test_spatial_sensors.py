"""Spatial invariance, local motion and sensory version compatibility."""
from dataclasses import replace
import numpy as np
import mujoco
import pytest
import torch
from flylab.arena import odor_sensors
from flylab.body import FlyBody
from flylab.senses import SensorSuite, SensorySettings
from flylab.simulation import Simulation
from flylab.live_state import load_live, save_live
from test_full_connectome import graph_fixture


def test_odor_uses_height_and_is_invariant_under_rigid_translation():
    body = FlyBody()
    source = body.data.xpos[body.body_ids['l_funiculus']].copy()
    close = odor_sensors(body, source, spatial_model='geometry-v3')
    high = source + [0,0,16]
    np.testing.assert_allclose(odor_sensors(body, high, spatial_model='geometry-v3'), close*np.exp(-2))
    np.testing.assert_array_equal(odor_sensors(body, high), odor_sensors(body, source))
    delta = np.array([8.,-3.,20.]); body.data.qpos[:3] += delta
    mujoco.mj_forward(body.model, body.data)
    np.testing.assert_allclose(odor_sensors(body, source+delta, spatial_model='geometry-v3'),close)


def test_rotating_body_generates_local_airflow_and_jacobian_matches_motion():
    body = FlyBody()
    settings = SensorySettings(spatial_model='geometry-v3', wind_enabled=True, wind_speed=0)
    suite = SensorSuite(settings)
    still = suite.sample(body, [0,0,0])
    body.data.qvel[5] = 20.
    before = body.data.qpos.copy()
    frame = suite.sample(body, [0,0,0])
    assert min(frame['wind']) > max(still['wind']) + .01
    legacy = SensorSuite(replace(settings, spatial_model='legacy-v2')).sample(body, [0,0,0])
    assert max(legacy['wind']) == 0
    bid = body.body_ids['l_funiculus']; origin = body.data.xpos[bid].copy()
    jac = np.zeros((3,body.model.nv)); mujoco.mj_jac(body.model,body.data,jac,None,origin,bid)
    velocity = jac @ body.data.qvel
    mujoco.mj_integratePos(body.model,body.data.qpos,body.data.qvel,1e-7)
    mujoco.mj_forward(body.model,body.data)
    np.testing.assert_allclose((body.data.xpos[bid]-origin)/1e-7,velocity,atol=2e-5)
    # The sensor itself did not advance any time.
    assert body.data.time == 0


def test_gravity_inversion_is_not_upright_and_zero_gravity_is_not_tilt():
    body = FlyBody(); settings = SensorySettings(spatial_model='geometry-v3',wind_enabled=True,wind_speed=0)
    suite = SensorSuite(settings)
    assert suite.sample(body,[0,0,0])['wind'] == [0.,0.]
    body.data.qpos[3:7] = [0.,1.,0.,0.]; mujoco.mj_forward(body.model,body.data)
    flipped = suite.sample(body,[0,0,0])
    np.testing.assert_allclose(flipped['gravity'],[0,0,1],atol=1e-10)
    np.testing.assert_allclose(flipped['wind'],[.5,.5])
    previous = body.model.opt.gravity.copy()
    try:
        body.model.opt.gravity[:] = 0
        zero = suite.sample(body,[0,0,0]); assert zero['wind'] == [0.,0.]
        assert zero['gravity'] == [0.,0.,0.]
    finally:
        body.model.opt.gravity[:] = previous


@pytest.mark.parametrize('dt',[0,-.01,float('nan'),float('inf')])
def test_invalid_sample_intervals_fail_without_mutation(dt):
    body=FlyBody(); before=body.data.qpos.copy()
    with pytest.raises(ValueError,match='Sensory interval'):
        SensorSuite().sample(body,[0,0,0],duration_s=dt)
    np.testing.assert_array_equal(body.data.qpos,before)


def test_upgrade_preserves_life_and_legacy_backup_and_exact_continuation(tmp_path,monkeypatch):
    import flylab.api as api
    graph_fixture(tmp_path); sim=Simulation(); sim.configure('connectome',tmp_path/'full'); sim.advance()
    monkeypatch.setattr(api,'sim',sim); monkeypatch.setattr(api,'DATA',tmp_path); monkeypatch.setattr(api,'ready',True)
    before=sim.full_brain.voltage.clone(); pose=sim.body.data.qpos.copy(); t=sim.time
    api.execute_control(api.Command(action='spatial_upgrade'))
    assert sim.time==t and not sim.running
    np.testing.assert_array_equal(pose,sim.body.data.qpos)
    torch.testing.assert_close(before,sim.full_brain.voltage,rtol=0,atol=0)
    backup=next((tmp_path/'sensory-backups').glob('*.pt'))
    assert load_live(backup,tmp_path/'full').sensors.settings.spatial_model=='legacy-v2'
    restored=load_live(tmp_path/'live-state.pt',tmp_path/'full')
    assert restored.sensors.settings.spatial_model=='geometry-v3'
    sim.advance(); restored.advance()
    np.testing.assert_array_equal(sim.body.data.qpos,restored.body.data.qpos)
    torch.testing.assert_close(sim.full_brain.voltage,restored.full_brain.voltage,rtol=0,atol=0)
    sim.environment_mode='spatial'; sim.source=np.array([0.,0.,30.])
    snapshot=sim.snapshot()
    assert snapshot['environment']['antennae']==snapshot['senses']['odor']
    assert snapshot['environment']['concentration']==pytest.approx(np.mean(snapshot['senses']['odor']))
