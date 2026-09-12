import torch
from flylab.training import Trainer, evaluate
from flylab.neural import ReferenceBrain
from flylab.simulation import Simulation

torch.set_num_threads(1)

def test_training_changes_only_plastic_parameters_and_improves_reward(tmp_path):
    brain=ReferenceBrain()
    original={k:v.clone() for k,v in brain.state_dict().items()}
    trainer=Trainer(tmp_path)
    trainer.start(brain,episodes=40,seed=42)
    trainer.thread.join(20)
    state=trainer.snapshot()
    assert not state['running'] and state['error'] is None
    assert state['after']['expected_reward'] > state['before']['expected_reward'] + .6
    assert state['parameter_change'] > .1
    for key,value in brain.state_dict().items(): assert torch.equal(value,original[key])
    assert torch.equal(trainer.result.fixed, original['fixed'])
    assert torch.equal(trainer.result.sensory, original['sensory'])
    sim=Simulation();trainer.apply(sim)
    assert evaluate(sim.brain)['expected_reward'] > .8
    loaded=Simulation();trainer.load(state['checkpoint'],loaded)
    assert torch.equal(loaded.brain.plastic,sim.brain.plastic)


def test_reversal_learning(tmp_path):
    trainer=Trainer(tmp_path);trainer.start(ReferenceBrain(),40,reward_odor='B',seed=2)
    trainer.thread.join(20)
    after=trainer.snapshot()['after']
    assert after['approach_B'] > .9 and after['approach_A'] < .1
