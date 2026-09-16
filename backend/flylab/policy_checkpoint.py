"""Iteration-boundary snapshots for the deterministic visual trainer.

Training randomness belongs to a local NumPy Generator. The policy has zero
dropout and no stochastic torch operations after its seeded initialization;
restoring process-global torch RNG would interfere with the live controller.
"""
import copy

import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def capture_training(model, optimizer, rng, *, global_step, validation_loss,
                     stale, early_stopping_reference_loss):
    return dict(state_dict=cpu_copy(model.state_dict()),
                optimizer_state_dict=cpu_copy(optimizer.state_dict()),
                rng_state=copy.deepcopy(rng.bit_generator.state),
                global_step=global_step, validation_loss=validation_loss,
                stale=stale, early_stopping_reference_loss=early_stopping_reference_loss)


def restore_training(snapshot, model, optimizer, rng, *, learning_rate, restore_rng=True):
    model.load_state_dict(snapshot['state_dict'], strict=True)
    # On CPU, load_state_dict may reuse tensor storage. Keep the saved best
    # snapshot immutable while the restored optimizer performs more updates.
    optimizer.load_state_dict(cpu_copy(snapshot['optimizer_state_dict']))
    # load_state_dict restores the old LR too. Apply the user's override last.
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    if restore_rng:
        rng.bit_generator.state = copy.deepcopy(snapshot['rng_state'])


def continuation_snapshot(saved, resume_from):
    if saved is None or 'training_state' not in saved:
        return None
    state = saved['training_state']
    if state.get('version') != 1:
        raise ValueError('Unsupported policy training checkpoint version')
    if resume_from not in {'latest', 'best'}:
        raise ValueError('Resume point must be latest or best')
    return state[resume_from]
