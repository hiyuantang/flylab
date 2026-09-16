"""Direct action-chunk behavior cloning with episode splits and physical evaluation.

The unchanged transformer learns the expert's actual excitation commands using
masked L1 regression. Physics is used to generate demonstrations and evaluate
closed-loop behavior; teacher muscle states are never supplied to the loss.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .body import BodyParameters, FlyBody
from .gesture_scene import GESTURES, random_placement
from .gesture_targets import muscle_reference
from .policy_model import SensorActionTransformer
from .policy_checkpoint import capture_training, continuation_snapshot, restore_training
from .policy_evaluation import evaluate_policy

RECIPE = 'action-chunk-bc-v1'


def action_loss(model, dataset, ids, device):
    observations = {key: value[ids].to(device) for key, value in dataset['observations'].items()}
    targets = dataset['actions'][ids].to(device)
    mask = dataset['mask'][ids].to(device)
    weights = dataset['weights'][ids].to(device)
    prediction = model(observations)
    # All 84 commands are supervised; future padding never contributes to loss.
    errors = (prediction - targets).abs().mean(-1)
    valid_weights = mask * weights[:, None]
    return (errors * valid_weights).sum() / valid_weights.sum()


def score_actions(model, dataset, device, batch_size=32):
    model.eval()
    numerator, denominator = 0., 0.
    with torch.no_grad():
        for start in range(0, len(dataset['actions']), batch_size):
            ids = slice(start, start + batch_size)
            weight = float((dataset['mask'][ids] * dataset['weights'][ids, None]).sum())
            numerator += float(action_loss(model, dataset, ids, device)) * weight
            denominator += weight
    return numerator / denominator


def build_dataset(trainer, settings, model, metadata, references):
    from .policy_training import teacher_examples, training_digest
    specification = dict(recipe=RECIPE, settings=settings, schema=model.schema(),
        seed=metadata['seed'], horizon=metadata['horizon'], proportions=metadata['proportions'],
        demonstrations_per_gesture=metadata['demonstrations_per_gesture'],
        validation_episodes=metadata['validation_episodes'], training_sha256=training_digest())
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    folder = trainer.directory / 'datasets'
    path = folder / f'{key}.pt'
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        return payload['splits'], key
    splits = {}
    for split, offset, count in [('train', 2000, metadata['demonstrations_per_gesture']),
                                 ('validation', 4000, metadata['validation_episodes'])]:
        observations = {s.name: [] for s in model.sensors}
        targets, masks, weights, episode_ids = [], [], [], []
        for episode in range(count):
            seed = metadata['seed'] + offset + episode
            for cue, reference in references.items():
                trainer.publish(phase=f'Preparing {split} demonstrations', current_gesture=cue,
                                batch_sample=episode + 1, sample_total=count)
                examples = teacher_examples(settings, reference, cue,
                    random_placement(np.random.default_rng(seed)), model.history, model.chunk, trainer.stop_event)
                for step, example in enumerate(examples):
                    for name in observations:
                        observations[name].append(example[0][name])
                    target = np.zeros((model.chunk, len(model.action_names)), dtype=np.float32)
                    target[:example[3]] = reference.commands[step:step + example[3], :len(model.action_names)]
                    targets.append(target)
                    masks.append(np.arange(model.chunk) < example[3])
                    weights.append(metadata['proportions'][cue] if split == 'train' else 1.)
                    episode_ids.append(f'{split}:{seed}:{cue}')
        splits[split] = dict(observations={k: torch.from_numpy(np.stack(v)) for k, v in observations.items()},
            actions=torch.from_numpy(np.stack(targets)), mask=torch.from_numpy(np.stack(masks)),
            weights=torch.tensor(weights, dtype=torch.float32), episode_ids=episode_ids)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(specification=specification, splits=splits), temporary)
    temporary.replace(path)
    return splits, key


def run_behavior_cloning(trainer, settings, saved, metadata):
    from .policy_training import policy_device, training_digest
    from .gesture_training import Cancelled
    started = time.perf_counter()
    device = policy_device()
    rng = np.random.default_rng(metadata['seed'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(metadata['seed'])
        body = FlyBody(BodyParameters(**settings['body']), 'lab')
        model = (SensorActionTransformer.from_schema(saved['schema']) if saved else
                 SensorActionTransformer(action_names=[body.model.actuator(i).name for i in range(84)]))
    selected = continuation_snapshot(saved, metadata['resume_from'])
    if saved:
        model.load_state_dict(selected['state_dict'] if selected else saved['state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=metadata['learning_rate'])
    resume = bool(selected and saved['metadata'].get('training_recipe') == RECIPE)
    if resume:
        restore_training(selected, model, optimizer, rng, learning_rate=metadata['learning_rate'],
                         restore_rng=metadata['seed'] == saved['metadata']['seed'])
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    metadata.update(trainable_parameters=sum(p.numel() for p in model.parameters()), schema=model.schema(),
                    training_sha256=training_digest())
    cues = [cue for cue in GESTURES if metadata['proportions'].get(cue, 0) > 0]
    references = {cue: muscle_reference(cue, metadata['horizon'], body.parameters, trainer.stop_event) for cue in cues}
    if not all(reference.evidence['pose_reached'] for reference in references.values()):
        raise ValueError('A physical teacher target was not reached; adjust the demonstration duration or body settings')
    trainer.publish(teacher_evaluation={g: r.evidence for g, r in references.items()})
    splits, dataset_key = build_dataset(trainer, settings, model, metadata, references)
    train, validation = splits['train'], splits['validation']
    signature = hashlib.sha256(json.dumps(dict(dataset=dataset_key, rollout_episodes=metadata['rollout_episodes'],
        rollout_horizon=metadata['horizon'], criterion='final-five-frames-v1'), sort_keys=True).encode()).hexdigest()
    same_validation = bool(resume and saved['metadata'].get('validation_signature') == signature)
    metadata.update(dataset_key=dataset_key, validation_signature=signature,
        training_windows=len(train['actions']), validation_windows=len(validation['actions']),
        validation_split='Separate placement seeds; complete trajectories stay in one split')
    baseline = score_actions(model, validation, device)
    global_step = metadata['parent_global_step']
    epoch_offset = selected.get('completed_epochs', 0) if resume else 0
    stale = selected['stale'] if same_validation else 0
    reference_loss = selected['early_stopping_reference_loss'] if same_validation else baseline

    def capture(loss, completed_epochs, evaluation=None):
        snapshot = capture_training(model, optimizer, rng, global_step=global_step,
            validation_loss=loss, stale=stale, early_stopping_reference_loss=reference_loss)
        snapshot.update(completed_epochs=completed_epochs, evaluation=evaluation)
        return snapshot

    latest = capture(baseline, epoch_offset, selected.get('evaluation') if same_validation else None)
    best = (saved['training_state']['best'] if same_validation and metadata['resume_from'] == 'latest' else
            selected if same_validation else latest)
    best_iteration = 0

    def selection(snapshot):
        evaluation = snapshot.get('evaluation')
        return (evaluation['success_rate'] if evaluation else -1., -snapshot['validation_loss'])

    def save(iteration):
        evaluation = best.get('evaluation')
        metadata.update(completed_iterations=iteration, best_iteration=best_iteration,
            global_step=global_step, completed_epochs=epoch_offset + iteration,
            best_global_step=best['global_step'], final_loss=best['validation_loss'],
            behavior_validated=bool(evaluation and evaluation['success_rate'] == 1.),
            selected_evaluation=evaluation, wall_seconds=time.perf_counter() - started)
        trainer._save(model, best['state_dict'], settings, metadata,
                      training_state=dict(version=1, latest=latest, best=best))

    trainer.publish(validation_initial_loss=baseline, validation_loss=baseline, validation_continued=same_validation,
        loss_metric='action_mae', training_recipe=RECIPE, training_windows=len(train['actions']),
        validation_windows=len(validation['actions']), global_step=global_step,
        adapter={'trainable_parameters': metadata['trainable_parameters'], 'parameter_change': 0.})
    save(0)
    previews = []
    for epoch in range(1, metadata['iterations'] + 1):
        if metadata['early_stopping'] and stale >= metadata['early_stopping_patience']:
            break
        started += trainer._pause_boundary()
        if trainer.stop_event.is_set():
            raise Cancelled()
        model.train()
        order = rng.permutation(len(train['actions']))
        numerator, denominator, norm = 0., 0., 0.
        for start in range(0, len(order), metadata['batch_size']):
            if trainer.stop_event.is_set():
                raise Cancelled()
            ids = order[start:start + metadata['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = action_loss(model, train, ids, device)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite action loss')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            global_step += 1
            weight = float((train['mask'][ids] * train['weights'][ids, None]).sum())
            numerator += float(loss.detach()) * weight
            denominator += weight
            trainer.publish(phase='Training action chunks', iteration=epoch, global_step=global_step,
                sample_step=min(start + len(ids), len(order)), sample_total=len(order), wall_seconds=time.perf_counter()-started)
        value = score_actions(model, validation, device)
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation action loss')
        meaningful = reference_loss - value > max(1e-8, reference_loss * .0001)
        stale = 0 if meaningful else stale + 1
        if meaningful:
            reference_loss = value
        evaluate = ((epoch_offset + epoch) % metadata['evaluation_interval'] == 0 or epoch == metadata['iterations']
                    or (metadata['early_stopping'] and stale >= metadata['early_stopping_patience']))
        evaluation = None
        if evaluate:
            def preview(cue, sim, count):
                frame = sim.snapshot()
                trainer.save_preview(metadata['id'], epoch, count, frame)
                previews.append(dict(epoch=epoch, sample=count, gesture=cue,
                    gesture_sample=(count - 1) % metadata['rollout_episodes'] + 1))
                trainer.publish(phase='Checking held-out physical gestures',
                    frame=frame, preview_iteration=epoch, preview_gesture=cue, previews=list(previews),
                    evaluation_episode=count, evaluation_total=len(cues) * metadata['rollout_episodes'],
                    wall_seconds=time.perf_counter()-started)
            evaluation = evaluate_policy(model, settings, cues,
                seeds=range(metadata['seed'] + 8000, metadata['seed'] + 8000 + metadata['rollout_episodes']),
                horizon=metadata['horizon'], stop=trainer.stop_event, publish=preview)
        latest = capture(value, epoch_offset + epoch, evaluation)
        if selection(latest) > selection(best):
            best, best_iteration = latest, epoch
        row = dict(iteration=epoch, global_step=global_step, loss=numerator / denominator,
            validation_loss=value, gradient_norm=norm, sample_losses=[], update_accepted=True,
            rollout_success_rate=evaluation['success_rate'] if evaluation else None,
            evaluation=evaluation)
        trainer.publish(history=trainer.status['history'] + [row], loss=row['loss'], validation_loss=value,
            best_validation_loss=best['validation_loss'], best_iteration=best_iteration, gradient_norm=norm,
            global_step=global_step, selected_evaluation=best.get('evaluation'),
            **({'evaluation': evaluation} if evaluation else {}),
            adapter={'trainable_parameters': metadata['trainable_parameters'],
                     'parameter_change': float((torch.nn.utils.parameters_to_vector(model.parameters()).detach()-initial).norm())})
        save(epoch)
    validated = metadata['behavior_validated']
    trainer.publish(running=False, resumable=False, stopping=False,
        phase=('Training complete · passed held-out gesture checks' if validated else
               'Training complete · gesture checks not passed'), behavior_validated=validated)
