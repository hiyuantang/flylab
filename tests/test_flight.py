import numpy as np
import pytest

from flylab.body import BodyParameters, compiled_model
from flylab.flight import FlightRig, FlightSettings, average_wingbeats, TRIM_FREQUENCY


def mean_force(samples):
    return np.array(average_wingbeats(samples, TRIM_FREQUENCY)['mean_force_uN'])


def test_native_air_has_lift_vacuum_has_none_and_legacy_model_is_unchanged():
    parameters = BodyParameters(appendage_model='peripheral-v2', elasticity_profile='stance-elastic-v1')
    original = compiled_model(parameters)
    before = (original.nu, original.opt.density, original.jnt_axis.copy(), original.body_mass.copy())
    air = FlightRig()
    samples = air.flap(.04)
    vacuum = FlightRig(FlightSettings(air_density=0., air_viscosity=0.))
    without_air = vacuum.flap(.04)
    assert mean_force(samples)[2] == pytest.approx(samples[-1]['weight_uN'], rel=.03)
    np.testing.assert_array_equal(mean_force(without_air), np.zeros(3))
    assert np.ptp([s['angles_rad'][1] for s in samples]) > 2.
    assert max(abs(t) for s in samples for t in s['servo_torque_uN_mm']) <= 30.
    assert air.model.nu == original.nu + 6
    assert original.nu == before[0] == 196
    assert original.opt.density == before[1] == 0
    np.testing.assert_array_equal(original.jnt_axis, before[2])
    np.testing.assert_array_equal(original.body_mass, before[3])
    assert air.mass_g == pytest.approx(original.body_mass.sum())
    assert np.max(np.abs(air.data.qpos[:3] - [0, 0, 10])) < .005


def test_halving_timestep_converges_for_reference_wingbeat():
    coarse, fine = FlightRig(), FlightRig(FlightSettings(timestep=.000025))
    a, b = coarse.flap(.04), fine.flap(.04)
    assert mean_force(a)[2] == pytest.approx(mean_force(b)[2], rel=.03)


def test_free_release_air_support_is_not_an_imposed_body_trajectory():
    air = FlightRig(FlightSettings(tethered=False))
    vacuum = FlightRig(FlightSettings(tethered=False, air_density=0., air_viscosity=0.))
    air.flap(.04)
    vacuum.flap(.04)
    assert air.data.eq_active[0] == vacuum.data.eq_active[0] == 0
    assert air.data.qpos[2] > vacuum.data.qpos[2] + 5
    # A trim reduces rotation but does not overwrite the free-body attitude.
    assert np.linalg.norm(air.data.qpos[4:7]) > 1e-4
    assert np.all(air.data.xfrc_applied == 0)
    assert np.all(air.data.qfrc_applied == 0)


def test_still_wings_do_not_generate_hovering_lift():
    rig = FlightRig()
    samples = rig.flap(.04, frequency=0, amplitude=0, feather=0)
    assert abs(mean_force(samples)[2]) < .01 * samples[-1]['weight_uN']


def test_input_limits_and_integer_physics_clock():
    with pytest.raises(ValueError):
        FlightSettings(air_density=-1)
    with pytest.raises(ValueError):
        FlightSettings(wind=(0, 0, float('nan')))
    rig = FlightRig()
    with pytest.raises(ValueError):
        rig.advance([0]*6, .000071)
    with pytest.raises(ValueError):
        rig.advance([2]*6, .001)
    with pytest.raises(ValueError):
        rig.flap(.04, frequency=float('nan'))
    rig.advance([0]*6, .001)
    assert rig.data.time == pytest.approx(.001)


def test_flight_api_is_bounded_and_does_not_dispatch_to_policy_training(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from flylab import flight_api
    app = FastAPI()
    app.include_router(flight_api.router)
    calls = []
    monkeypatch.setattr(flight_api, 'run_experiment', lambda request: calls.append(request) or {'isolated': True})
    client = TestClient(app)
    assert client.post('/api/flight/experiment', json={}).json() == {'isolated': True}
    assert calls[-1].tethered
    assert client.post('/api/flight/experiment', json={'frequency': 1000}).status_code == 422
    assert client.post('/api/flight/experiment', json={'duration': 1000}).status_code == 422
    flight_api.experiment_lock.acquire()
    try:
        assert client.post('/api/flight/experiment', json={}).status_code == 409
    finally:
        flight_api.experiment_lock.release()


def test_calibrated_wingbeat_balances_force_and_com_moment():
    rig = FlightRig()
    means = average_wingbeats(rig.flap(.08), TRIM_FREQUENCY)
    assert means['averaging_cycles'] == 11
    assert abs(means['mean_force_uN'][0]) < .03
    assert means['mean_force_uN'][2] == pytest.approx(rig.mass_g*9810, abs=.03)
    assert abs(means['mean_torque_uN_mm'][1]) < .03
    assert rig.model.joint('l_wing_spread').id < rig.model.joint('l_wing_elevation').id


def test_torque_is_about_center_of_mass_and_uses_force_state():
    import mujoco
    for quaternion in ([1, 0, 0, 0], [2**-.5, 0, 2**-.5, 0]):
        rig = FlightRig()
        rig.data.qpos[3:7] = quaternion
        mujoco.mj_forward(rig.model, rig.data)
        force = np.array([0., 0., 2.])
        rig.data.qfrc_fluid[:] = 0
        # Use MuJoCo's actual Jacobian mapping: a force at the center of mass
        # has zero moment about that center, including for a rotated free root.
        mujoco.mj_applyFT(rig.model, rig.data, force, np.zeros(3),
                         rig.data.subtree_com[rig.root_id], rig.root_id, rig.data.qfrc_fluid)
        np.testing.assert_allclose(rig.measure()['fluid_torque_uN_mm'], [0, 0, 0], atol=1e-12)
        expected = np.cross(rig.data.subtree_com[rig.root_id]-rig.data.xpos[rig.root_id], force)
        np.testing.assert_allclose(rig.measure()['fluid_root_torque_uN_mm'], expected, atol=1e-12)


def test_partial_wingbeat_does_not_bias_mean_force():
    h, frequency = .00001, 275.
    samples = [dict(time=(i+1)*h, force_time_s=i*h,
                    fluid_force_uN=[0, 0, np.sin(2*np.pi*frequency*i*h)],
                    fluid_torque_uN_mm=[0, 0, 0], servo_power_uW=0.) for i in range(4250)]
    result = average_wingbeats(samples, frequency)
    assert result['averaging_cycles'] == 5
    assert abs(result['mean_force_uN'][2]) < .001
    assert abs(np.mean([s['fluid_force_uN'][2] for s in samples[2125:]])) > .01


def test_solver_warning_aborts_even_if_state_remains_finite(monkeypatch):
    import mujoco
    rig = FlightRig()
    def warning_step(model, data, nstep):
        data.time += nstep*model.opt.timestep
        data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1
    monkeypatch.setattr(mujoco, 'mj_step', warning_step)
    with pytest.raises(RuntimeError, match='solver'):
        rig.advance([0]*6, .00005)


def test_hover_assist_sustains_free_flight_and_cannot_hover_in_vacuum():
    from flylab.flight_control import free_rollout
    rig = FlightRig()
    samples = free_rollout(rig, 1.)
    assert max(abs(s['position_mm'][2]-10) for s in samples) < 1
    assert min(s['upright'] for s in samples) > np.cos(np.deg2rad(15))
    assert max(np.linalg.norm(s['position_mm'][:2]) for s in samples) < 5
    assert not any(s['environment_contacts'] or s['tethered'] for s in samples)
    assert rig.data.time == pytest.approx(1.08)
    assert max(abs(t) for s in samples for t in s['servo_torque_uN_mm']) <= 30
    assert not np.any(rig.data.xfrc_applied) and not np.any(rig.data.qfrc_applied)
    vacuum = FlightRig(FlightSettings(air_density=0., air_viscosity=0.))
    falling = free_rollout(vacuum, .08)
    assert min(s['position_mm'][2] for s in falling) < 5
    assert not np.any([s['fluid_force_uN'] for s in falling])


def test_cancelled_api_request_keeps_lock_until_worker_finishes(monkeypatch):
    import asyncio
    import threading
    from flylab import flight_api
    started, release = threading.Event(), threading.Event()
    def blocked(_):
        started.set()
        release.wait(5)
        return {}
    monkeypatch.setattr(flight_api, 'run_experiment', blocked)
    async def scenario():
        task = asyncio.create_task(flight_api.experiment(flight_api.FlightRequest()))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert flight_api.experiment_lock.locked()
        finally:
            release.set()
            await asyncio.gather(*flight_api.pending_experiments)
        assert not flight_api.experiment_lock.locked()
    asyncio.run(scenario())
