from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flylab.matrix_readout import MatrixMotorReadout
from flylab.motor_mapping import PERIPHERAL_PROFILE


@pytest.mark.parametrize('device', ['cpu','mps'])
def test_matrix_preserves_weighted_routes_empty_channels_and_dynamic_gains(device):
    if device=='mps' and not torch.backends.mps.is_available(): pytest.skip('Apple GPU unavailable')
    bridge=SimpleNamespace(mapping_profile=PERIPHERAL_PROFILE,
        channels=[torch.tensor([1,3]),torch.tensor([],dtype=torch.long),torch.tensor([3])],
        channel_weights=[torch.tensor([.5,1.],dtype=torch.float64),torch.tensor([]),torch.tensor([.25])],
        brain=SimpleNamespace(rates=torch.tensor([900.,20.,500.,80.],device=device)),
        parameters=np.zeros(8,dtype=np.float32))
    readout=MatrixMotorReadout(bridge,device)
    assert readout.matrix.shape==(3,2)  # Only mapped inputs; every route retained.
    np.testing.assert_allclose(readout(),[.45,0,.2],atol=1e-7)
    bridge.parameters[7]=np.log(2)
    np.testing.assert_allclose(readout(),[.9,0,.4],atol=1e-7)
    bridge.brain.rates[3]=400
    np.testing.assert_allclose(readout(),[1,0,1],atol=1e-7)
