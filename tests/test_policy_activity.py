"""Telemetry measures real forwards without changing policy weights or outputs."""
import torch
from flylab.policy_model import SensorActionTransformer, SensorSpec
from flylab.policy_activity import PolicyActivity


def test_activity_values_and_inference_isolation():
    torch.manual_seed(7)
    model = SensorActionTransformer([SensorSpec('touch', 3)], ['left', 'right'], history=2,
                                   chunk=3, width=16, layers=2, heads=2).eval()
    observation = {'touch': torch.rand(1, 2, 3, 1)}
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        expected = model(observation)
    activity = PolicyActivity(model)
    assert activity.snapshot(0)['values'] == {}
    with torch.inference_mode():
        actual = model(observation)
        tokens = model.tokenizers['touch'](observation['touch'])
    torch.testing.assert_close(actual, expected)
    result = activity.snapshot(1)
    assert result['measured']
    assert result['values']['muscles'] == actual[0, 0].tolist()
    torch.testing.assert_close(torch.tensor(result['values']['touch']), tokens[0].square().mean(-1).sqrt())
    for group in result['groups']:
        assert len(result['values'][group['id']]) == group['count']
        assert torch.isfinite(torch.tensor(result['values'][group['id']])).all()
    assert activity.snapshot(0)['values'] == {}  # reset clears displayed trial
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    revision = activity.revision
    model.train()
    model(observation).sum().backward()
    assert activity.revision == revision  # telemetry does not retain training graphs
    assert all(p.grad is not None for p in model.parameters())
