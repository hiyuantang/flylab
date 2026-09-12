import numpy as np
import torch
from flylab.simulation import Simulation


def test_silencing_motor_stops_command_then_muscles_decay():
    sim = Simulation()
    sim.advance(30)
    active = np.mean(sim.body.data.act)
    sim.silenced.add('motor')
    sim.advance(30)
    assert torch.count_nonzero(sim.state[84:]) == 0
    assert np.mean(sim.body.data.act) < active*.01
    assert np.isfinite(sim.body.data.qpos).all()


def test_stimulus_is_local_and_expires():
    a, b = Simulation(), Simulation()
    b.stimulate('optic', amplitude=2, duration=.1)
    a.advance(3); b.advance(3)
    assert b.state[:16].mean() > a.state[:16].mean() + .2
    b.advance(5)
    assert not next(r for r in b.snapshot()['regions'] if r['id']=='optic')['stimulated']


def test_body_moves_from_muscle_forces_and_reset_preserves_learning():
    sim = Simulation()
    initial = sim.body.data.qpos.copy()
    sim.advance(50)
    assert np.linalg.norm(sim.body.data.qpos[7:] - initial[7:]) > .01
    assert np.max(np.abs(sim.body.data.actuator_force)) > 0
    with torch.no_grad(): sim.brain.plastic.add_(.1)
    learned = sim.brain.plastic.detach().clone()
    sim.reset()
    assert torch.equal(learned, sim.brain.plastic)
    assert sim.steps == 0 and sim.time == 0
    assert torch.count_nonzero(sim.state) == 0


def test_reference_is_deterministic():
    a,b=Simulation(),Simulation()
    a.advance(10);b.advance(10)
    np.testing.assert_array_equal(a.body.data.qpos,b.body.data.qpos)
    assert torch.equal(a.state,b.state)
