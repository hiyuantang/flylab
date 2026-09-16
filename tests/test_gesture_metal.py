from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from flylab.full_connectome import FullBrain, Physiology
from flylab.connectome_adapter import ConnectomeAdapter
from flylab.neuromuscular import NeuromuscularBridge
from flylab.gesture_metal import MetalGradientBrain
from flylab.gesture_gradients import Spike
from test_full_connectome import graph_fixture

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Real Apple GPU required')


def make_engine(tmp_path, delay, scale=1.):
    graph_fixture(tmp_path)
    brain = FullBrain(tmp_path / 'full', Physiology.paper(delay_ms=delay), device='mps', precision='float16')
    adapter = ConnectomeAdapter(brain, rank=3)
    engine = MetalGradientBrain(brain, adapter, NeuromuscularBridge(brain), surrogate_scale=scale)
    engine.sync_weights()
    return engine


@pytest.mark.parametrize('scale', [1., .1, .01, .001, .0003])
@pytest.mark.parametrize('delay', [0., 1.8])
def test_fp16_forward_matches_existing_gpu_inference(tmp_path, delay, scale):
    engine = make_engine(tmp_path, delay, scale)
    b = engine.brain
    drive = torch.tensor([15., 3., 0.])
    b.advance(drive, 20.)
    names = ('voltage','current','spikes','rates','delay_queue','last_spike_tick','spike_counts')
    expected = {name:getattr(b,name).cpu().clone() for name in names}
    b.reset(); engine.reset()
    rates = engine.advance(drive)
    for name in names:
        torch.testing.assert_close(getattr(b,name).cpu(), expected[name], atol=0, rtol=0)
    rates.sum().backward()
    assert torch.isfinite(engine.parameters.grad).all()
    assert engine.parameters.grad.norm() > 0
    base = engine.adapter.base.clone()
    before = engine.parameters.detach().clone()
    torch.optim.Adam([engine.parameters], lr=.01).step()
    assert not torch.equal(engine.parameters, before)
    engine.detach_state(); engine.sync_weights()
    torch.testing.assert_close(engine.adapter.base, base, atol=0, rtol=0)
    assert not b.weights.requires_grad


@pytest.mark.parametrize('delay', [0., 1.8])
@pytest.mark.parametrize('scale', [1., .1, .01, .001, .0003])
@pytest.mark.parametrize('full_history', [False, True])
@pytest.mark.parametrize('stable_history', [False, True])
def test_fp32_backward_matches_independent_dense_autograd(tmp_path, delay, full_history, stable_history, scale):
    engine = make_engine(tmp_path, delay, scale)
    b = engine.brain
    engine.full_history = full_history
    engine.stable_history = stable_history and full_history
    raw_history, flags_history = [], []
    recording = True
    lib = engine.lib
    def capture(*args, **kwargs):
        lib.integrate(*args, **kwargs)
        if recording:
            raw_history.append(args[9].clone()); flags_history.append(args[10].clone())
    engine.lib = SimpleNamespace(integrate=capture, finish=lib.finish, incoming_grad=lib.incoming_grad,
        transpose_grad=lib.transpose_grad, factor_grad=lib.factor_grad, active_factor_grad=lib.active_factor_grad, reduce_factors=lib.reduce_factors,
        reverse_state=lib.reverse_state)
    drive = torch.tensor([15., 3., 0.])
    upstream = torch.tensor([.3, -.7, 1.2])
    if full_history:
        engine.advance(drive)
    objective = (engine.advance(drive)*upstream.to('mps')).sum()
    recording = False
    forward_state = {name:getattr(b,name).clone() for name in ('voltage','current','rates','delay_queue','last_spike_tick','spike_counts')}
    if engine.stable_history:
        from flylab.gesture_history import backward_full_history
        backward_full_history(engine, objective)
    else:
        objective.backward()
    for name,value in forward_state.items():
        torch.testing.assert_close(getattr(b,name),value,atol=0,rtol=0)
    actual = engine.parameters.grad.cpu() * (2. ** getattr(engine, 'gradient_exponent', 0))
    # Independent dense PyTorch graph, with observed FP16 threshold voltages
    # and fixed refractory/reset choices. Rounding uses a straight-through VJP.
    raw_history = [x.cpu().float() for x in raw_history]
    flags_history = [x.cpu() for x in flags_history]
    params = engine.parameters.detach().cpu().requires_grad_()
    u,v = params.unbind()
    factors = 1+.75*torch.tanh(u@v.T/math.sqrt(engine.adapter.rank))
    effective = engine.adapter.base.float()*factors.flatten()[torch.from_numpy(engine.adapter.pairs).long()]
    effective = effective+(b.weights.values().half().float()-effective).detach()
    dense = torch.zeros(3,3)
    posts = torch.repeat_interleave(torch.arange(3), b.weights.crow_indices().diff())
    dense = dense.index_put((posts,b.weights.col_indices()),effective)
    voltage, current, rates = torch.full((3,),-52.), torch.zeros(3), torch.zeros(3)
    queue = [torch.zeros(3) for _ in range(len(b.delay_queue))]
    p = engine.coefficients.cpu().float()
    for t,(observed, flags) in enumerate(zip(raw_history,flags_history)):
        free, fired = (flags&1)!=0, (flags&2)!=0
        raw = torch.where(free, -52+(voltage+52)*p[0]+current*p[2]+drive*p[3], voltage)
        raw = raw+(observed-raw).detach()
        spikes = Spike.apply(raw+45, scale)*free
        slot, future = t%len(queue), (t+b.delay_steps)%len(queue)
        queue[future] = queue[future]+spikes
        incoming = dense@queue[slot]
        queue[slot] = torch.zeros(3)
        voltage = torch.where(fired,-52.,raw)
        current = torch.where(fired,0.,torch.where(free,current*p[1],current)+incoming*(free&~fired))
        rates = rates*p[4]+spikes*p[5]
    (rates*upstream).sum().backward()
    torch.testing.assert_close(actual, params.grad, atol=2e-5, rtol=2e-4)


def test_chunked_factor_reduction_keeps_large_and_empty_groups():
    from flylab.gesture_metal import training_kernels
    rng = np.random.default_rng(9)
    counts = [2500, 0, 17, 1200]
    bounds = np.r_[0,np.cumsum(counts)]
    chunks = [np.arange(a,b,1024) for a,b in zip(bounds[:-1],bounds[1:])]
    ptr = np.r_[np.concatenate(chunks),bounds[-1]]
    group_ptr = np.r_[0,np.cumsum([len(x) for x in chunks])]
    pre,post = rng.integers(0,100,(2,sum(counts)))
    base = rng.normal(size=sum(counts)).astype('float32')
    emitted = rng.normal(size=100).astype('float16')
    di = rng.normal(size=100).astype('float32')
    gpu = lambda x,dtype:torch.tensor(x,device='mps',dtype=dtype)
    lib = training_kernels()
    partial = torch.empty(len(ptr)-1,device='mps')
    result = torch.zeros(len(counts),device='mps')
    lib.factor_grad(gpu(ptr,torch.int32),gpu(pre,torch.int32),gpu(post,torch.int32),
        gpu(base,torch.float32),gpu(emitted,torch.float16),gpu(di,torch.float32),partial,
        threads=len(partial)*32,group_size=256)
    lib.reduce_factors(gpu(group_ptr,torch.int32),partial,result,threads=len(counts)*32,group_size=256)
    terms = base*emitted[pre].astype('float32')*di[post]
    expected = np.array([terms[a:b].sum(dtype='float64') for a,b in zip(bounds[:-1],bounds[1:])])
    np.testing.assert_allclose(result.cpu(),expected,atol=2e-5,rtol=2e-5)


def test_gpu_checkpoint_records_precision_and_reloads_for_inference(tmp_path):
    from flylab.gesture_batch import BatchGestureTrainer
    from flylab.gesture_training import GestureSession
    from test_gesture_batch import run
    from test_gesture_training import settings
    graph_fixture(tmp_path)
    trainer = BatchGestureTrainer(tmp_path/'weights',tmp_path/'full')
    result = run(trainer,proportions={'point':1})
    saved = trainer._load(result['checkpoint'])
    assert result['execution'] == {'device':'mps','precision':'float16','gradient_precision':'float32'}
    assert saved['metadata']['execution'] == result['execution']
    assert saved['parameters'].device.type == 'cpu' and saved['parameters'].dtype == torch.float32
    restored = GestureSession(tmp_path/'full',saved['settings'],saved['rank'],saved['seed'])
    restored.adapter.apply(saved['parameters'].numpy())
    assert restored.sim.full_brain.device.type == 'mps'
    assert restored.sim.full_brain.dtype == torch.float16
    restored.sim.advance()
    assert not restored.sim.full_brain.rates.requires_grad


def test_active_factor_segments_skip_only_exact_zero_emissions(tmp_path):
    e=make_engine(tmp_path,1.8)
    emitted=torch.tensor([0.,2.,0.],device='mps',dtype=torch.float16)
    di=torch.tensor([.3,-.7,1.2],device='mps')
    partial=torch.empty(e.pair_chunks,device='mps')
    actual=torch.zeros(e.adapter.group_count**2,device='mps')
    e.lib.active_factor_grad(e.pair_ptr,e.pair_source,e.pair_post,e.pair_base,emitted,di,
                            partial,threads=e.pair_chunks*32,group_size=256)
    e.lib.reduce_factors(e.group_ptr,partial,actual,threads=actual.numel()*32,group_size=256)
    expected=torch.zeros_like(actual,device='cpu')
    for pre in range(len(e.brain.ids)):
        for edge in range(int(e.brain.out_ptr[pre]),int(e.brain.out_ptr[pre+1])):
            expected[e.adapter.out_pairs[edge]] += float(emitted[pre])*float(e.adapter.base_out[edge])*float(di[e.brain.out_post[edge]])
    torch.testing.assert_close(actual.cpu(),expected,atol=2e-5,rtol=2e-5)
