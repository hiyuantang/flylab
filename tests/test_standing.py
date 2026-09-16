"""Weight support must distinguish a standing fly from one resting on its body."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from flylab.body import FlyBody, BodyParameters
from flylab.senses import SensorSuite, SensorySettings
from flylab.simulation import Simulation
from flylab.live_state import save_live, load_live
from test_full_connectome import graph_fixture


def test_collapsed_upright_body_is_not_foot_supported():
    body = FlyBody(BodyParameters(appendage_model='peripheral-v1'))
    for _ in range(120):
        body.step_muscles(np.zeros(body.model.nu), dt=1/60)
    support = body.support_snapshot()
    assert body.snapshot()['upright'] > .9
    assert not support['feet_supported']
    assert support['other_vertical_uN'] > support['weight_uN'] / 2
    assert body.legacy_leg_feedback().sum() > body.foot_feedback().sum()
    assert 'c_thorax' in support['other_contacts']
    assert sum(support['foot_vertical_uN']) + sum(support['other_contacts'].values()) == pytest.approx(support['world_force_uN'][2])
    legacy = SensorSuite().sample(body, [0,0,0])
    corrected = SensorSuite(SensorySettings(contact_model='tarsal-contact-v1')).sample(body, [0,0,0])
    np.testing.assert_allclose(legacy['touch'], np.clip(body.legacy_leg_feedback()/5, 0, 1))
    np.testing.assert_allclose(corrected['touch'], np.clip(body.foot_feedback()/5, 0, 1))


def test_elastic_baseline_support_is_passive_and_unpinned():
    body = FlyBody(BodyParameters(appendage_model='peripheral-v1', elasticity_profile='stance-elastic-v1'))
    start = body.data.qpos[:3].copy()
    for _ in range(120):
        body.step_muscles(np.zeros(body.model.nu), dt=1/60)
    support = body.support_snapshot()
    assert support['feet_supported']
    assert support['supporting_feet'] == 6
    assert support['other_vertical_uN'] == 0
    assert sum(support['foot_vertical_uN']) == pytest.approx(support['weight_uN'], rel=.1)
    assert not np.any(body.data.ctrl) and not np.any(body.data.act)
    assert not np.any(body.data.xfrc_applied) and not np.any(body.data.qfrc_applied)
    assert np.linalg.norm(body.data.qpos[:3] - start) > .01
    assert body.model.neq == 0  # No tether or fixed-root equality.


def test_standing_trial_preserves_graph_and_checkpoint_continuation(tmp_path):
    graph_fixture(tmp_path)
    sim = Simulation(); sim.configure('connectome', tmp_path/'full')
    brain = sim.full_brain; ids = brain.ids.copy(); weights = brain.weights.values().clone()
    sim.prepare_standing_trial()
    assert sim.full_brain is brain and sim.time == 0 and not sim.running
    np.testing.assert_array_equal(brain.ids, ids)
    torch.testing.assert_close(brain.weights.values(), weights, rtol=0, atol=0)
    assert sim.odor == 'none' and sim.sensors.settings.contact_model == 'tarsal-contact-v1'
    assert sim.sensors.settings.stimulus_position == tuple(sim.body.scene['sound_source'])
    assert sim.bridge.parameters[7] == -2
    sim.advance(); path = tmp_path/'standing.pt'; save_live(sim, path)
    restored = load_live(path, tmp_path/'full')
    sim.advance(); restored.advance()
    np.testing.assert_array_equal(sim.body.data.qpos, restored.body.data.qpos)
    torch.testing.assert_close(sim.full_brain.voltage, restored.full_brain.voltage, rtol=0, atol=0)


def test_wall_normals_are_not_vertical_support_and_geom_order_is_respected(monkeypatch):
    from types import SimpleNamespace as NS
    import mujoco
    body = object.__new__(FlyBody)
    body.parameters = BodyParameters()
    body.body_ids = {'c_thorax': 1}
    body.model = NS(geom_bodyid=np.array([0, 2]), body_mass=np.array([0., .001]),
                    opt=NS(gravity=np.array([0.,0.,-9810.])), body=lambda _: NS(name='lf_tarsus5'))
    body.data = NS(xmat=np.tile(np.eye(3).ravel(), (3,1)), contact=[
        NS(geom1=0, geom2=1, frame=np.eye(3).ravel()),
        NS(geom1=1, geom2=0, frame=np.diag([-1.,1.,-1.]).ravel())])
    def contact_force(model, data, index, result):
        result[:] = [5., 0., 2., 0., 0., 0.]
    monkeypatch.setattr(mujoco, 'mj_contactForce', contact_force)
    support = body.support_snapshot()
    assert support['foot_normal_uN'][0] == 10
    assert support['foot_vertical_uN'][0] == 4  # Wall friction; not 10 µN normal load.
    assert support['world_force_uN'] == [10., 0., 4.]
    assert not support['feet_supported']


def test_standing_control_backs_up_the_previous_life(tmp_path, monkeypatch):
    import flylab.api as api
    graph_fixture(tmp_path)
    sim = Simulation(); sim.configure('connectome', tmp_path/'full'); sim.advance()
    before = sim.time
    monkeypatch.setattr(api, 'sim', sim); monkeypatch.setattr(api, 'DATA', tmp_path)
    monkeypatch.setattr(api, 'ready', True)
    api.execute_control(api.Command(action='standing_trial'))
    backup = next((tmp_path/'standing-backups').glob('*.pt'))
    previous = load_live(backup, tmp_path/'full')
    assert previous.time == before
    assert previous.body.parameters.elasticity_profile == 'baseline'
    assert sim.time == 0 and sim.body.parameters.elasticity_profile == 'stance-elastic-v1'
