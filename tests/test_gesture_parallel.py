from threading import Event
import numpy as np
import pytest
import torch
from flylab.gesture_parallel import ParallelMetalBrain
from test_gesture_metal import make_engine

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Real Apple GPU required')


@pytest.mark.parametrize('size',[1,2,3])
@pytest.mark.parametrize('delay',[0.,1.8])
def test_parallel_states_and_averaged_gradients_match_sequential(tmp_path,size,delay):
    e=make_engine(tmp_path,delay)
    drives=torch.tensor([[15.,3.,0.],[0.,17.,2.],[20.,0.,9.]])[:size]
    masks=torch.tensor([[False,False,False],[True,False,False],[False,False,True]])[:size]
    expected=[]
    upstream=torch.tensor([.3,-.7,1.2],device='mps')
    e.parameters.grad=None
    for i in range(size):
        e.brain.reset();e.reset()
        for _ in range(2):
            (e.advance(drives[i],silence=masks[i])*upstream).sum().div(size).backward()
            e.detach_state()
        expected.append({name:getattr(e.brain,name).clone() for name in ParallelMetalBrain.STATE_NAMES})
        expected[-1]['delay_queue']=e.brain.delay_queue.clone()
    gradient=e.parameters.grad.clone()
    e.parameters.grad=None;e.brain.reset();e.reset()
    parallel=ParallelMetalBrain(e,size)
    # There is one shared model/adapter; lanes allocate state only.
    assert parallel.shared is e and parallel.shared.parameters is e.parameters
    for _ in range(2):
        (parallel.advance(drives,silence=masks)*upstream).sum().div(size).backward()
        parallel.detach_state()
    for i in range(size):
        for name in ParallelMetalBrain.STATE_NAMES:
            torch.testing.assert_close(getattr(parallel,name)[i],expected[i][name],atol=0,rtol=0)
        torch.testing.assert_close(parallel.delay_queue[:,i,:],expected[i]['delay_queue'],atol=0,rtol=0)
    torch.testing.assert_close(e.parameters.grad,gradient,atol=2e-5,rtol=3e-4)
    parallel.export_sample(size-1)
    for name in ParallelMetalBrain.STATE_NAMES:
        torch.testing.assert_close(getattr(e.brain,name),expected[-1][name],atol=0,rtol=0)
    assert e.brain.tick_index==400


def test_parallel_bodies_losses_preview_and_checkpoint(tmp_path):
    from flylab.gesture_batch import BatchGestureTrainer
    from test_full_connectome import graph_fixture
    from test_gesture_batch import settings, fixture_demonstrations
    graph_fixture(tmp_path)
    trainer=BatchGestureTrainer(tmp_path/'weights',tmp_path/'full',reference_factory=fixture_demonstrations)
    frames=[]
    publish=trainer.publish
    def capture(**values):
        if 'frame' in values:frames.append(values['preview_iteration'])
        publish(**values)
    trainer.publish=capture
    trainer.start(settings(),iterations=2,horizon=5,batch_size=3,
                  proportions={'point':1,'point_right':1,'point_both':1})
    trainer.thread.join(45)
    status=trainer.snapshot()
    assert not status['running'] and not status['error']
    assert status['parallel_samples']==3 and status['completed_rollouts']==6
    assert frames==[1,2]
    assert set(status['batch_gestures'])=={'point','point_right','point_both'}
    assert status['frame']['gesture_stimulus']['gesture']==status['last_batch'][-1]['gesture']
    assert status['frame']['gesture_stimulus']['position']==status['last_batch'][-1]['placement']['position']
    assert status['loss']==pytest.approx(np.mean([x['loss'] for x in status['last_batch']]))
    saved=trainer._load(status['checkpoint'])
    assert saved['metadata']['batch_mode']=='parallel'
    assert len(saved['results']['history'])==2
    assert all(len(row['sample_losses'])==3 for row in saved['results']['history'])


def test_parallel_cancel_does_not_save_partial_batch(tmp_path):
    from flylab.gesture_batch import BatchGestureTrainer
    from test_full_connectome import graph_fixture
    from test_gesture_batch import settings, fixture_demonstrations
    graph_fixture(tmp_path)
    trainer=BatchGestureTrainer(tmp_path/'weights',tmp_path/'full',reference_factory=fixture_demonstrations)
    publish=trainer.publish
    def cancel(**values):
        publish(**values)
        if values.get('sample_step')==1:trainer.stop_event.set()
    trainer.publish=cancel
    trainer.start(settings(),iterations=1,horizon=5,batch_size=3)
    trainer.thread.join(30)
    status=trainer.snapshot()
    assert not status['running'] and status['cancelled']
    assert not status['checkpoint'] and not status['history'] and not trainer.checkpoints()


def test_parallel_extended_body_has_independent_activation_state(tmp_path):
    from flylab.gesture_batch import BatchSession
    from flylab.gesture_scene import random_placement
    from flylab.motor_mapping import EXTENDED_PROFILE
    from test_full_connectome import graph_fixture
    from test_gesture_batch import settings, fixture_demonstrations
    graph_fixture(tmp_path)
    config=settings()
    config['mapping_profile']=EXTENDED_PROFILE
    config['body']['appendage_model']='peripheral-v2'
    config['training_execution']={'device':'mps','precision':'float16'}
    session=BatchSession(tmp_path/'full',config,2,42)
    assert session.sim.body.model.nu==196
    cues=['point','point_right','point_both']
    rng=np.random.default_rng(42)
    samples,frame,comparison=session.parallel_samples(cues,[random_placement(rng) for _ in cues],
        {cue:torch.zeros((5,196),dtype=torch.float64) for cue in cues},Event())
    assert len(samples)==3 and len(comparison['actual'])==84
    pointers=[body.data.act.__array_interface__['data'][0] for body in session._parallel_bodies]
    assert len(set(pointers))==3
    assert frame['gesture_stimulus']['gesture']=='point_both'


def test_training_and_live_gpu_work_share_serial_execution(tmp_path):
    from threading import RLock, Thread
    from flylab.gesture_batch import BatchSession
    from flylab.gesture_scene import random_placement
    from test_gesture_batch import settings
    from test_full_connectome import graph_fixture
    graph_fixture(tmp_path)
    config = settings()
    config['training_execution'] = {'device':'mps', 'precision':'float16'}
    session = BatchSession(tmp_path/'full', config, 2, 42)
    (tmp_path/'live').mkdir()
    live = make_engine(tmp_path/'live', 1.8)
    lock = RLock()
    session.execution_lock = lock
    stopped, advanced = Event(), Event()
    errors, snapshots = [], []
    def inference():
        try:
            while not stopped.is_set():
                with lock:
                    live.brain.advance(torch.tensor([15.,3.,0.]), 1.)
                    torch.mps.synchronize()
                    snapshots.append(live.brain.rates.cpu().clone())
                advanced.set()
                stopped.wait(.001)
        except BaseException as exc:
            errors.append(exc)
    worker = Thread(target=inference)
    worker.start()
    try:
        assert advanced.wait(10)
        cues = ['point', 'point_right', 'point']
        targets = {cue:torch.zeros((5, session.sim.body.model.nu)) for cue in cues}
        samples, _, _ = session.parallel_samples(cues,
            [random_placement(np.random.default_rng(i)) for i in range(3)], targets, Event())
        assert len(samples) == 3 and all(np.isfinite(s['loss']) for s in samples)
    finally:
        stopped.set()
        worker.join(10)
    assert not worker.is_alive() and not errors and snapshots


@pytest.mark.parametrize('scale', [1., .001, .0003])
@pytest.mark.parametrize('stable', [False, True])
@pytest.mark.parametrize('delay', [0., 1.8])
def test_full_history_batch_matches_independent_sample_gradients(tmp_path, delay, stable, scale):
    e = make_engine(tmp_path, delay, scale)
    e.full_history = True
    drives = torch.tensor([[15.,3.,0.],[0.,17.,2.]])
    upstream = torch.tensor([.3,-.7,1.2],device='mps')
    expected = []
    for i in range(2):
        e.brain.reset(); e.reset()
        loss = 0.
        for _ in range(3):
            loss = loss + (e.advance(drives[i])*upstream).sum()/2
        loss.backward()
        expected.append(e.brain.rates.clone())
        e.detach_state()
    gradient = e.parameters.grad.clone()
    e.parameters.grad = None
    e.brain.reset();e.reset()
    e.stable_history = stable
    parallel = ParallelMetalBrain(e, 2)
    loss = 0.
    for _ in range(3):
        loss = loss + (parallel.advance(drives)*upstream).sum()/2
    before = parallel.rates.clone()
    if stable:
        from flylab.gesture_history import backward_full_history, scale_power_two
        backward_full_history(parallel, loss)
        e.parameters.grad = scale_power_two(e.parameters.grad, e.gradient_exponent)
    else:
        loss.backward()
    torch.testing.assert_close(parallel.rates,before,atol=0,rtol=0)
    for i in range(2):
        torch.testing.assert_close(parallel.rates[i],expected[i],atol=0,rtol=0)
    torch.testing.assert_close(e.parameters.grad,gradient,atol=2e-5,rtol=3e-4)


@pytest.mark.parametrize('batched', [False, True])
def test_validation_without_tape_preserves_forward_state(tmp_path, batched):
    engine = make_engine(tmp_path, 1.8)
    model = ParallelMetalBrain(engine, 2) if batched else engine
    brain = model if batched else engine.brain
    drive = torch.tensor([[15., 3., 0.], [0., 17., 2.]]) if batched else torch.tensor([15., 3., 0.])
    for _ in range(3):
        model.advance(drive)
    names = ('voltage', 'current', 'rates', 'delay_queue', 'spike_counts', 'last_spike_tick')
    expected = {name: getattr(brain, name).clone() for name in names}
    engine.brain.reset(); model.reset()
    with torch.no_grad():
        for _ in range(3):
            model.advance(drive)
    for name, value in expected.items():
        torch.testing.assert_close(getattr(brain, name), value, atol=0, rtol=0)


def test_preview_precedes_backward_and_reports_each_checkpoint(tmp_path, monkeypatch):
    from flylab import gesture_history
    from flylab.gesture_batch import BatchSession
    from flylab.gesture_scene import random_placement
    from test_full_connectome import graph_fixture
    from test_gesture_batch import settings
    graph_fixture(tmp_path)
    config = settings()
    config['training_execution'] = {'device': 'mps', 'precision': 'float16'}
    session = BatchSession(tmp_path/'full', config, 2, 42)
    events = []
    backward = gesture_history.backward_full_history
    def inspect(*args, **kwargs):
        assert len([event for event in events if 'frame' in event]) == 1
        assert session.gradient.parameters.grad is None
        return backward(*args, **kwargs)
    monkeypatch.setattr(gesture_history, 'backward_full_history', inspect)
    samples, frame, _ = session.parallel_samples(['point', 'point_right'],
        [random_placement(np.random.default_rng(i)) for i in range(2)],
        {cue: torch.zeros((5, session.sim.body.model.nu)) for cue in ['point', 'point_right']},
        Event(), progress=lambda **event: events.append(event))
    previews = [event for event in events if 'frame' in event]
    assert previews[0]['frame'] is frame
    assert previews[0]['last_batch'] == samples
    assert [event['backward_step'] for event in events if 'backward_step' in event] == list(range(6))
    assert torch.isfinite(session.gradient.parameters.grad).all()
