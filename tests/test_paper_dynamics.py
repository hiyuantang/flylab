import numpy as np
import pytest
import torch
from flylab.full_connectome import Physiology
from flylab.paper_dynamics import PaperDynamics


class TinyBrain(PaperDynamics):
    def __init__(self, weights, **parameters):
        self.ids = np.arange(len(weights))
        self.config = Physiology.paper(**parameters)
        self.weights = torch.tensor(weights, dtype=torch.float64).to_sparse_csr()
        self.prepare_paper()
        self.reset_paper()


@pytest.mark.parametrize('delay_ms', [0., 1.8])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_pytorch_matches_brian2_linear_delay_refractory_schedule(delay_ms, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU unavailable')
    import brian2 as b
    b.start_scope()
    b.prefs.codegen.target = 'numpy'
    b.defaultclock.dt = .1 * b.ms
    # Simultaneous excitation/inhibition, recurrent feedback, and a disconnected
    # neuron test threshold, delay, reset and writes during refractory periods.
    weights = np.array([[0, 0, 0, 0], [50, 0, -8, 0], [45, 30, 0, 0], [0, 0, 0, 0.]])
    brain = TinyBrain(weights, delay_ms=delay_ms)
    brain.set_device(device)
    group = b.NeuronGroup(4, 'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\ndg/dt=-g/(5*ms):volt (unless refractory)',
                          threshold='v > -45*mV', reset='v=-52*mV;g=0*mV', refractory=2.2*b.ms, method='linear')
    group.v = -52 * b.mV
    syn = b.Synapses(group, group, 'w:volt', on_pre='g += w', delay=delay_ms*b.ms)
    post, pre = weights.nonzero()
    syn.connect(i=pre, j=post)
    syn.w = weights[post, pre] * b.mV
    events = np.zeros((500, 4))
    events[[0, 1, 20, 23, 70, 130, 200, 350], 0] = 68.75
    events[[40, 150], 2] = 10
    # A jump at threshold must NOT fire (the source uses >, not >=).
    events[0, 3] = 7
    @b.network_operation(dt=.1*b.ms, when='synapses', order=1)
    def stimulus(t):
        step = round(float(t / b.ms) * 10)
        group.v[:] += events[step] * np.asarray(group.not_refractory) * b.mV
    states = b.StateMonitor(group, ['v', 'g'], record=True, when='end')
    spikes = b.SpikeMonitor(group)
    b.Network(group, syn, states, spikes, stimulus).run(50*b.ms)
    v, g, spike_trace = [], [], []
    for row in events:
        brain.advance_paper(torch.zeros(4), .1, voltage_events=torch.tensor(row[None, :]))
        v.append(brain.voltage.cpu().clone().numpy())
        g.append(brain.current.cpu().clone().numpy())
        spike_trace.append(brain.spikes.cpu().clone().numpy())
    tolerance = 5e-5 if device == 'mps' else 2e-11
    np.testing.assert_allclose(v, np.asarray(states.v / b.mV).T, rtol=0, atol=tolerance)
    np.testing.assert_allclose(g, np.asarray(states.g / b.mV).T, rtol=0, atol=tolerance)
    expected = np.zeros((500, 4))
    expected[np.rint(np.asarray(spikes.t/b.ms)*10).astype(int), np.asarray(spikes.i)] = 1
    np.testing.assert_array_equal(spike_trace, expected)
    assert expected[:, 1].sum() > 0 and expected[:, 3].sum() == 0


def test_event_delivery_is_full_sparse_matrix_product_and_batching_is_invariant():
    weights = [[0, 1, -2], [5, 0, 0], [0, 3, 0]]
    brain = TinyBrain(weights)
    emitted = torch.tensor([.3, 2., 1.], dtype=torch.float64)
    torch.testing.assert_close(brain.deliver(emitted), brain.weights @ emitted, rtol=0, atol=0)
    zero = torch.zeros(3)
    rates = torch.tensor([200., 0., 0.])
    brain.refractory_ticks[0] = 0
    brain.advance_paper(zero, 20, poisson_hz=rates)
    other = TinyBrain(weights)
    other.refractory_ticks[0] = 0
    for _ in range(20):
        other.advance_paper(zero, 1, poisson_hz=rates)
    for key in ['voltage', 'current', 'rates', 'spike_counts', 'delay_queue']:
        torch.testing.assert_close(getattr(brain, key), getattr(other, key), rtol=0, atol=0)
