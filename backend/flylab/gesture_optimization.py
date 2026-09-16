"""Measured validation safeguards for approximate spike gradients.

Validation uses fixed visual inputs; it is a model-selection check, not an
independent generalization test. Targets and the forward brain are unchanged.
"""
import copy
import numpy as np
import torch


def adam_step(optimizer, parameters, evaluate, previous_loss):
    """Retain finite Adam updates even when validation worsens."""
    original = parameters.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    accepted = False
    try:
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise ValueError('Nonfinite or missing adapter gradient')
        optimizer.step()
        if not torch.isfinite(parameters).all() or any(
                not torch.isfinite(value).all()
                for values in optimizer.state.values() for value in values.values()
                if torch.is_tensor(value)):
            raise ValueError('Nonfinite Adam update')
        with torch.no_grad():
            parameters.clamp_(-4, 4)
        loss = float(evaluate())
        if not np.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        accepted = True
        return {'validation_loss': loss, 'update_accepted': True,
                'update_scale': 1., 'validation_trials': [{'scale': 1., 'loss': loss}]}
    finally:
        if not accepted:
            with torch.no_grad():
                parameters.copy_(original)
            optimizer.load_state_dict(state)


class BestValidation:
    """Track exact best weights separately from meaningful-improvement patience."""
    def __init__(self, parameters, loss, patience=10):
        if not np.isfinite(loss):
            raise ValueError('Nonfinite initial validation loss')
        self.parameters = parameters.detach().clone()
        self.loss = self.reference_loss = loss
        self.iteration = self.stalled = 0
        self.patience = patience

    def observe(self, parameters, loss, iteration):
        if not np.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        if loss < self.loss:
            self.loss, self.iteration = loss, iteration
            self.parameters.copy_(parameters.detach())
        if loss < self.reference_loss - max(1e-8, abs(self.reference_loss) * 1e-4):
            self.reference_loss, self.stalled = loss, 0
        else:
            self.stalled += 1
        return self.stalled >= self.patience


def target_reachability(session, targets, proportions):
    """An exact lower bound from zero-output channels; not a full reachability proof."""
    body, bridge = session.sim.body, session.sim.bridge
    missing = [i for i in range(84) if not len(bridge.channels[i])]
    total = sum(proportions.values())
    floors = {cue: float(values[:, missing].square().sum() / (len(values) * 84))
              for cue, values in targets.items() if proportions.get(cue, 0) > 0}
    floor = sum(proportions[cue] / total * value for cue, value in floors.items())
    return {'target_channels': 84, 'unrouted_channels': len(missing),
            'unrouted_names': [body.model.actuator(i).name for i in missing],
            'unrouted_target_channels': sum(any(float(targets[cue][:, i].abs().max()) > 1e-8 for cue in floors) for i in missing),
            'unavoidable_mse': floor, 'per_gesture_floor': floors,
            'full_target_reachable': False if floor > 1e-10 else None,
            'note': 'Zero-output muscle channels create this loss floor. Routed channels may have further limits.'}
