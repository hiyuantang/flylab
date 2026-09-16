import numpy as np
import pytest
import torch
from flylab.full_connectome import FullBrain
from flylab.cpg_rate import RateExperiment
from test_full_connectome import graph_fixture


def test_rate_equation_orientation_and_full_graph_kept(tmp_path):
    graph_fixture(tmp_path)
    brain = FullBrain(tmp_path / 'full')
    original = brain.weights.clone()
    r = RateExperiment(brain, np.array([1.,2.,4.]))
    assert r.weights.shape == (3,3) and r.weights._nnz() == original._nnz()
    # The fixture has seven measured ACh synapses 0 -> 1.
    state, drive = torch.tensor([100.,0,0]), torch.zeros(3)
    incoming = torch.tensor([0.,2.1e1,0.])
    expected = (torch.clamp(r.cap*torch.tanh(r.slope/r.cap*(incoming-r.threshold)), min=0)-state)/r.tau
    torch.testing.assert_close(r.derivative(state, drive), expected)
    assert torch.equal(original.values(), brain.weights.values())
    assert r.metadata['volume_missing'] == 0


def test_rate_step_convergence_and_no_self_generated_activity_without_input(tmp_path):
    graph_fixture(tmp_path); brain=FullBrain(tmp_path/'full')
    volumes=np.array([1.,2.,np.nan])
    fine=RateExperiment(brain,volumes,dt_ms=.25)
    coarse=RateExperiment(brain,volumes,dt_ms=.5)
    zero=torch.zeros(3)
    coarse.advance(zero,10); assert not coarse.rates.any()
    drive=torch.tensor([100.,0,0])
    fine.advance(drive,100);coarse.advance(drive,100)
    torch.testing.assert_close(fine.rates,coarse.rates,atol=.003,rtol=.001)
    assert fine.rates[0]>0 and fine.rates[1]>0 and fine.rates[2]==0
    assert fine.metadata['volume_missing']==1
    with pytest.raises(ValueError): coarse.advance(drive,.6)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_rate_gpu_cpu_equivalence(tmp_path):
    graph_fixture(tmp_path);brain=FullBrain(tmp_path/'full')
    volumes=np.array([1.,2.,4.])
    cpu=RateExperiment(brain,volumes)
    gpu=RateExperiment(brain,volumes,device='mps')
    drive=torch.tensor([100.,0,0])
    cpu.advance(drive,100);gpu.advance(drive,100)
    torch.testing.assert_close(cpu.rates,gpu.rates.cpu(),rtol=2e-5,atol=1e-5)
