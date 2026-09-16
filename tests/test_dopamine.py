import math
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch

from flylab.full_connectome import build_full, FullBrain, Physiology
from flylab.dopamine import PROFILE, MIN_FACTOR
from flylab.simulation import Simulation
from flylab.live_state import save_live, load_live, NEURAL_TENSORS


def reward_graph(path):
    # Reward and punishment share one KC, with a separate KC lacking DA input.
    types = ['PAM01', 'PPL101', 'KCg-m', 'KCg-m', 'MBON01', 'MBON11', 'PAM01']
    nts = ['dopamine', 'dopamine', 'acetylcholine', 'acetylcholine', 'glutamate', 'gaba', 'unclear']
    rows = [{'bodyId': i + 10, 'type': t, 'superclass': 'cb_intrinsic',
             'class': 'Kenyon_Cell' if t.startswith('KC') else 'DAN' if 'P' in t else 'MBON',
             'rootSide': 'L', 'somaSide': 'L', 'subclass': None} for i, t in enumerate(types)]
    feather.write_feather(pa.Table.from_pylist(rows), path/'body-annotations.feather')
    edges = [(10,12,5),(11,12,10),(10,14,1),(11,15,1),
             (12,14,8),(12,15,12),(13,14,9),(16,13,9),(16,14,1)]
    feather.write_feather(pa.table(dict(zip(['body_pre','body_post','weight'],zip(*edges)))),path/'connectome-weights.feather')
    feather.write_feather(pa.table({'body':list(range(10,17)),'consensus_nt':nts,'predicted_nt_confidence':[.9]*7}),path/'body-neurotransmitters.feather')
    build_full(path)
    return path/'full'


def make_brain(path, device='cpu'):
    return FullBrain(reward_graph(path), Physiology.paper(dt_ms=1.,timing_rounding='ceil'),
                     device=device,precision='float16' if device=='mps' else 'float64')


def test_mapping_valence_and_bounded_simulation_time_pulses(tmp_path):
    brain=make_brain(tmp_path)
    reward=brain.reward_circuit
    assert reward.available and len(reward.edges)==2
    assert [len(d) for d in reward.dans]==[1,1]  # Unclear transmitter is excluded.
    assert reward.summary()['profile']==PROFILE
    for score, index in [(1,0),(-.5,1)]:
        reward.submit(score,0)
        rate=reward.input_rates(.02)
        assert rate.nonzero().flatten().tolist()==[index]
        assert rate[index]==120*abs(score)
        with pytest.raises(ValueError,match='still active'):reward.submit(score,0)
        for _ in range(10):reward.update(torch.zeros(7),.02)
        assert reward.input_rates(.02) is None and reward.remaining==0
    reward.submit(0,.4)
    assert reward.events==2
    for invalid in [math.nan,math.inf,1.01,-1.01]:
        with pytest.raises(ValueError):reward.submit(invalid,0)
    reward.submit(1,.4)
    reward.update(torch.zeros(7),.19)
    assert reward.input_rates(.02)[0]==pytest.approx(60.)
    reward.stop()
    assert not reward.enabled and reward.input_rates(.02) is None


@pytest.mark.parametrize('channel',[0,1])
def test_learning_requires_recent_kc_and_measured_dan_activity(tmp_path,channel):
    brain=make_brain(tmp_path);reward=brain.reward_circuit
    base=brain.weights.values().clone()
    # Disabled teaching still observes recent KC activity, without weight changes.
    rates=torch.zeros(7);rates[2:4]=100
    reward.update(rates,.5)
    assert np.all(reward.factors==1)
    reward.submit(1 if channel==0 else -1,.5)
    reward.update(rates,.1)
    assert np.all(reward.factors==1)  # Score alone never directly changes weights.
    rates[channel]=100
    reward.update(rates,.2)
    assert reward.factors[channel]<1 and reward.factors[1-channel]==1
    torch.testing.assert_close(brain.weights.values(),base,rtol=0,atol=0)
    torch.testing.assert_close(brain.out_weight[reward.outgoing],reward.base*torch.from_numpy(reward.factors))
    # An edge from the KC without measured DAN input cannot be modified.
    pos=brain.out_ptr[3].item()
    assert brain.out_weight[pos]==9*brain.config.gain
    before=reward.factors.copy();reward.stop();reward.update(rates,.5)
    np.testing.assert_array_equal(reward.factors,before)
    reward.enabled=True
    for _ in range(100):
        reward.remaining=.2
        reward.update(rates,.2)
    assert reward.factors.min()==MIN_FACTOR
    brain.reset()
    assert not reward.enabled and np.all(reward.factors==1)
    torch.testing.assert_close(brain.out_weight[reward.outgoing],reward.base)


def test_inactive_kc_and_dan_output_silencing_block_learning(tmp_path):
    brain=make_brain(tmp_path);reward=brain.reward_circuit;reward.submit(1,0)
    rates=torch.zeros(7);rates[0]=100
    reward.update(rates,.2)
    assert np.all(reward.factors==1)
    rates[2]=100
    brain.output_gain[0]=0
    reward.release.fill(0)
    reward.update(rates,.2)
    assert np.all(reward.factors==1)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_dan_spikes_drive_learning_and_checkpoint_continuation(tmp_path,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Apple GPU unavailable')
    directory=reward_graph(tmp_path)
    sim=Simulation()
    sim.configure('connectome',directory,Physiology.paper(dt_ms=1000/960,timing_rounding='ceil'),
                  device=device,precision='float16' if device=='mps' else 'float64',command_hz=60,coupling_mode='pipelined')
    reward=sim.full_brain.reward_circuit
    # Supply recent sensory eligibility; subsequent DA must actually fire.
    reward.eligibility.fill(1)
    reward.submit(1,0)
    sim.advance(4)
    assert sim.full_brain.spike_counts[0]>0
    assert reward.factors[0]<1 and reward.factors[1]==1
    path=tmp_path/'live.pt';save_live(sim,path)
    assert torch.load(path,weights_only=True)['format']=='flylab-live-v3'
    restored=load_live(path,directory)
    for _ in range(3):
        sim.advance();restored.advance()
        for key in NEURAL_TENSORS:
            torch.testing.assert_close(getattr(sim.full_brain,key),getattr(restored.full_brain,key),rtol=0,atol=0)
        np.testing.assert_array_equal(sim.full_brain.reward_circuit.factors,restored.full_brain.reward_circuit.factors)
        np.testing.assert_array_equal(sim.body.data.qpos,restored.body.data.qpos)
    assert restored.full_brain.reward_circuit.summary()==reward.summary()
    payload=torch.load(path,weights_only=True)
    payload['dopamine']['factors'][0]=0
    torch.save(payload,path)
    with pytest.raises(ValueError,match='dopamine'):load_live(path,directory)


def test_cpu_effective_delivery_and_device_migration(tmp_path):
    brain=make_brain(tmp_path);reward=brain.reward_circuit
    events=torch.zeros(7,dtype=torch.float64);events[2]=1
    before=brain.deliver(events).clone()
    reward.factors[:]=[.5,.8];reward.apply_weights()
    after=brain.deliver(events)
    assert after[4]==before[4]*.5 and after[5]==before[5]*.8
    if torch.backends.mps.is_available():
        brain.set_device('mps','float16')
        torch.testing.assert_close(brain.metal.weight[reward.edges.to('mps')].cpu(),(reward.base*torch.from_numpy(reward.factors)).half())
        brain.set_device('cpu','float64')
        torch.testing.assert_close(brain.deliver(events),after,rtol=0,atol=0)


def test_no_matching_anatomy_fails_explicitly(tmp_path):
    from test_full_connectome import graph_fixture
    graph_fixture(tmp_path)
    reward=FullBrain(tmp_path/'full').reward_circuit
    assert not reward.available
    with pytest.raises(ValueError,match='annotated'):reward.submit(1,0)


def test_tonic_firing_other_valence_and_expired_pulse_are_not_new_rewards(tmp_path):
    brain=make_brain(tmp_path);reward=brain.reward_circuit
    brain.rates[:]=100
    reward.update(brain.rates.cpu(),.5)
    reward.submit(1,0)
    reward.update(brain.rates.cpu(),.1)
    assert np.all(reward.factors==1)  # Constant pre-feedback baseline.
    brain.rates[1]=200
    reward.update(brain.rates.cpu(),.05)
    assert np.all(reward.factors==1)  # Unselected punishment population.
    brain.rates[0]=200
    reward.update(brain.rates.cpu(),.05)
    assert reward.factors[0]<1 and reward.factors[1]==1
    assert reward.remaining==0
    old=reward.release.copy()
    reward.update(brain.rates.cpu(),.1)
    np.testing.assert_allclose(reward.release,old*np.exp(-.1/.2))


def test_invalid_learning_state_cannot_overwrite_saved_state(tmp_path):
    directory=reward_graph(tmp_path)
    sim=Simulation();sim.configure('connectome',directory)
    reward=sim.full_brain.reward_circuit
    path=tmp_path/'life.pt';save_live(sim,path);before=path.read_bytes()
    reward.factors[0]=float('nan')
    with pytest.raises(ValueError,match='dopamine'):save_live(sim,path)
    assert path.read_bytes()==before
