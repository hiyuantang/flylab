from dataclasses import replace
import threading
import numpy as np
import pytest
import torch
from test_full_connectome import graph_fixture
from flylab.coupling import run_pair
from flylab.simulation import Simulation
from flylab.live_state import save_live, load_live, NEURAL_TENSORS


def test_pair_runs_concurrently_and_joins_on_brain_failure():
    brain_started, body_started, body_done = (threading.Event() for _ in range(3))
    def brain():
        brain_started.set()
        assert body_started.wait(1)
        raise ValueError('brain failed')
    def body():
        body_started.set()
        assert brain_started.wait(1)
        body_done.set()
    with pytest.raises(ValueError, match='brain failed'):
        run_pair(brain, body)
    assert body_done.is_set()


def test_pair_propagates_physics_failure_after_brain_finishes():
    finished = threading.Event()
    def brain():
        finished.set()
        return 42
    def body():
        raise ValueError('body failed')
    with pytest.raises(ValueError, match='body failed'):
        run_pair(brain, body)
    assert finished.is_set()


def make_sim(tmp_path, device='cpu'):
    graph_fixture(tmp_path)
    sim = Simulation()
    sim.configure('connectome', tmp_path / 'full', device=device,
                  precision='float16' if device=='mps' else 'float64')
    sim.set_command_hz(60)
    sim.set_coupling('pipelined')
    return sim


def test_body_uses_previous_command_and_samples_before_motion(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)
    prior = np.full(sim.body.model.nu, .2, dtype=np.float32)
    newer = np.full(sim.body.model.nu, .7, dtype=np.float32)
    sim.pending_command = prior.copy()
    original = sim.body.step_muscles
    controls = []
    def step(command, dt):
        controls.append(command.copy())
        original(command, dt)
    monkeypatch.setattr(sim.body, 'step_muscles', step)
    monkeypatch.setattr(sim.bridge, 'muscles', lambda: newer.copy())
    sim.advance()
    np.testing.assert_array_equal(controls[0], prior)
    np.testing.assert_array_equal(sim.pending_command, newer)
    assert sim.last_sensory_time == 0 and sim.command_generated_at == sim.time
    assert sim.applied_command_generated_at == 0
    sim.advance()
    np.testing.assert_array_equal(controls[1], newer)
    assert sim.last_sensory_time == pytest.approx(1/60)
    assert sim.coupling_summary()['command_delay_ms'] == pytest.approx(1000/60)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_parallel_matches_same_delay_serial_and_exact_checkpoint(tmp_path, device, monkeypatch):
    if device=='mps' and not torch.backends.mps.is_available(): pytest.skip('Apple GPU unavailable')
    import flylab.simulation as module
    sim = make_sim(tmp_path, device)
    sim.stimulate('antennal', 3, .5)
    sim.advance(2)
    path = tmp_path / 'pipeline.pt'
    save_live(sim,path)
    assert torch.load(path,weights_only=True)['format']=='flylab-live-v2'
    restored = load_live(path,tmp_path/'full')
    assert restored.coupling_mode=='pipelined'
    for _ in range(4):
        sim.advance()
        with monkeypatch.context() as patch:
            patch.setattr(module,'run_pair',lambda brain, body:run_pair(brain,body,concurrent=False))
            restored.advance()
        for key in NEURAL_TENSORS:
            torch.testing.assert_close(getattr(sim.full_brain,key),getattr(restored.full_brain,key),rtol=0,atol=0)
        np.testing.assert_array_equal(sim.body.data.qpos,restored.body.data.qpos)
        np.testing.assert_array_equal(sim.body.data.qvel,restored.body.data.qvel)
        np.testing.assert_array_equal(sim.pending_command,restored.pending_command)
        assert sim.coupling_summary()==restored.coupling_summary()
        assert sim.time==restored.time and sim.body.data.time==restored.body.data.time
    sim.set_command_hz(50)
    sim.advance()
    save_live(sim,path)
    restored=load_live(path,tmp_path/'full')
    assert restored.command_hz==50 and restored.coupling_summary()['command_delay_ms']==20
    sim.configure('connectome',physiology=replace(sim.full_brain.config,gain=.3))
    assert sim.coupling_mode=='pipelined' and sim.time==0
    assert not sim.pending_command.any()


def test_failed_cycle_joins_and_cannot_overwrite_checkpoint(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)
    path=tmp_path/'life.pt'
    save_live(sim,path)
    original=path.read_bytes()
    def fail(*args,**kwargs):raise RuntimeError('brain failed')
    monkeypatch.setattr(sim.bridge,'advance_command',fail)
    with pytest.raises(RuntimeError,match='brain failed'):sim.advance()
    assert sim.cycle_incomplete
    body_time=sim.body.data.time
    assert body_time==pytest.approx(1/60)
    with pytest.raises(ValueError,match='incomplete'):save_live(sim,path)
    assert path.read_bytes()==original
    with pytest.raises(RuntimeError,match='incomplete'):sim.advance()
    sim.reset()
    assert not sim.cycle_incomplete and sim.time==0


@pytest.mark.parametrize('field,value',[('generated_at',-1),('mode','bad'),('command',torch.tensor([1.])),('source_time',100)])
def test_invalid_pipeline_checkpoint_rejected(tmp_path,field,value):
    sim=make_sim(tmp_path)
    path=tmp_path/'life.pt'
    save_live(sim,path)
    payload=torch.load(path,weights_only=True)
    payload['coupling'][field]=value
    torch.save(payload,path)
    with pytest.raises(ValueError):load_live(path,tmp_path/'full')


@pytest.mark.parametrize('field,value', [('pending_command',np.array([np.nan],dtype=np.float32)), ('command_generated_at',100.)])
def test_invalid_buffer_cannot_replace_checkpoint(tmp_path,field,value):
    sim=make_sim(tmp_path)
    path=tmp_path/'life.pt'
    save_live(sim,path)
    before=path.read_bytes()
    setattr(sim,field,value)
    with pytest.raises(ValueError,match='buffered'):save_live(sim,path)
    assert path.read_bytes()==before


def test_configure_rejects_invalid_coupling_before_mutation(tmp_path):
    sim=make_sim(tmp_path)
    body=sim.body
    for controller,mode in [('connectome','unknown'),('synthetic','pipelined')]:
        with pytest.raises(ValueError):sim.configure(controller,coupling_mode=mode)
        assert sim.body is body and sim.coupling_mode=='pipelined'
