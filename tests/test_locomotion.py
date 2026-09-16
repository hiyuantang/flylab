from dataclasses import replace
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
import pyarrow.feather as feather
from flylab.body import FlyBody, LEGS
from flylab.leg_feedback import LegFeedback, sample_legs, PROFILE
from flylab.locomotion import LocomotorCommand
from flylab.senses import SensorSuite, SensorySettings
from flylab.live_state import save_live, load_live, NEURAL_TENSORS
from flylab.simulation import Simulation
from test_full_connectome import graph_fixture


@pytest.mark.parametrize('leg', LEGS)
def test_signed_tibia_velocity_matches_physical_geometry(leg):
    body = FlyBody()
    joint = body.model.joint(leg.lower() + '_tibia_pitch')
    qi, vi = joint.qposadr[0], joint.dofadr[0]
    body.data.qvel[vi] = 2.
    before = body.data.qpos.copy()
    measured = sample_legs(body)
    np.testing.assert_array_equal(body.data.qpos, before)
    i = LEGS.index(leg)
    body.data.qpos[qi] += 2e-6
    mujoco.mj_forward(body.model, body.data)
    finite_difference = (sample_legs(body)['opening_rad'][i] - measured['opening_rad'][i]) / 1e-6
    assert measured['velocity_rad_s'][i] == pytest.approx(finite_difference, abs=1e-4)
    assert measured['signals']['flexing'][i] > 0
    assert measured['signals']['extending'][i] == 0
    body.data.qvel[vi] = -2.
    reverse = sample_legs(body)
    assert reverse['signals']['extending'][i] > 0 and reverse['signals']['flexing'][i] == 0
    off = sample_legs(body, False)
    assert not np.any(list(off['signals'].values()))


def test_type_side_nerve_specific_feedback_and_unresolved_not_guessed():
    def row(t, side='L', nerve='ProLN'):
        return dict(type=t, rootSide=side, entryNerve=nerve,
                    superclass='vnc_sensory', **{'class': 'mechanosensory_proprioceptive'})
    rows = [row('SNpp50'), row('SNpp51'), row('SNpp39'), row('SNpp41'),
            row('SNpp39', 'R'), row('SNpp50', nerve='MesoLN'),
            row('SNpp40'), row('unknown'), row('SNpp50', side=None)]
    routing = LegFeedback(rows)
    signals = {k: [0.] * 6 for k in ['flexed', 'extended', 'extending', 'flexing']}
    signals['flexed'][0] = .8
    signals['extended'][0] = .2
    signals['extending'][0] = .6
    drive = torch.zeros(len(rows))
    routing.apply(drive, {'profile': PROFILE, 'signals': signals}, 1.)
    torch.testing.assert_close(drive, torch.tensor([1.6, .4, 1.2, 0., 0., 0., 0., 0., 0.]))
    assert routing.summary()['unresolved_proprioceptors'] == 3


def test_walk_drive_targets_only_named_dns_expires_and_stops():
    brain = SimpleNamespace(neurons=[{'type': t} for t in ['DNg100', 'IN17A001', 'DNb08', 'Ti flexor MN']],
                            ids=np.array([1,2,3,4]), rates=torch.zeros(4))
    c = LocomotorCommand(brain)
    c.submit('DNg100', 1.2, .1, 0.)
    drive = torch.zeros(4); c.apply(drive, .099)
    torch.testing.assert_close(drive, torch.tensor([1.2,0,0,0]))
    drive.zero_(); c.apply(drive, .1); assert not drive.any()
    c.submit('DNb08', 2., 1., .1); c.reset()
    c.apply(drive, .1); assert not drive.any()
    for target, amp in [('motor', 1), ('DNg100', float('nan')), ('DNg100', 4)]:
        with pytest.raises(ValueError): c.submit(target, amp, 1., 0.)


def test_walk_and_feedback_checkpoint_exact_continuation(tmp_path):
    graph_fixture(tmp_path)
    file = tmp_path / 'full' / 'neurons.feather'
    table = feather.read_table(file)
    types = table['type'].to_pylist(); types[0] = 'DNg100'
    import pyarrow as pa
    feather.write_feather(table.set_column(table.schema.get_field_index('type'), 'type', pa.array(types)), file)
    sim = Simulation(); sim.configure('connectome', tmp_path / 'full')
    sim.sensors = SensorSuite(replace(sim.sensors.settings, proprioception_model=PROFILE))
    sim.bridge.locomotion.submit('DNg100', 2., .5, 0.)
    sim.advance()
    file = tmp_path / 'walking.pt'; save_live(sim, file)
    assert torch.load(file, weights_only=True)['format'] == 'flylab-live-v4'
    restored = load_live(file, tmp_path / 'full')
    assert restored.sensors.settings.proprioception_model == PROFILE
    assert restored.bridge.locomotion.state_dict() == sim.bridge.locomotion.state_dict()
    for _ in range(3):
        sim.advance(); restored.advance()
        for name in NEURAL_TENSORS:
            torch.testing.assert_close(getattr(sim.full_brain, name), getattr(restored.full_brain, name), rtol=0, atol=0)
        np.testing.assert_array_equal(sim.body.data.qpos, restored.body.data.qpos)
    sim.reset(); assert sim.bridge.locomotion.amplitude == 0


def test_sensor_profiles_keep_legacy_behavior_and_expose_measured_kinematics():
    body = FlyBody()
    legacy = SensorSuite().sample(body, [0,0,0])
    assert legacy['legs'] is None
    assert SensorySettings(**legacy['settings']).proprioception_model == 'legacy-position-v1'
    new = SensorSuite(SensorySettings(proprioception_model=PROFILE)).sample(body, [0,0,0])
    assert len(new['legs']['opening_rad']) == 6
    assert new['legs']['signals']['extending'] == [0.] * 6


def test_proprioception_upgrade_preserves_brain_and_backups(tmp_path, monkeypatch):
    import flylab.api as api
    graph_fixture(tmp_path)
    sim = Simulation(); sim.configure('connectome', tmp_path / 'full'); sim.advance()
    monkeypatch.setattr(api, 'sim', sim); monkeypatch.setattr(api, 'DATA', tmp_path); monkeypatch.setattr(api, 'ready', True)
    time = sim.time; voltage = sim.full_brain.voltage.clone()
    api.execute_control(api.Command(action='proprioception_upgrade'))
    assert sim.time == time and sim.sensors.settings.proprioception_model == PROFILE
    torch.testing.assert_close(sim.full_brain.voltage, voltage, rtol=0, atol=0)
    backup = next((tmp_path / 'sensory-backups').glob('*.pt'))
    assert load_live(backup, tmp_path / 'full').sensors.settings.proprioception_model == 'legacy-position-v1'
    assert load_live(tmp_path / 'live-state.pt', tmp_path / 'full').sensors.settings.proprioception_model == PROFILE
