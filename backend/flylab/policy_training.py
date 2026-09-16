"""Visual-policy training, checkpoint orchestration, and teacher trajectory tools.

The current action-chunk behavior-cloning loop lives in policy_bc. The legacy
activation-loss helper remains available for historical diagnostics only.
Inference and physical evaluation use vision without teacher state.
"""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import threading
import time

import numpy as np
import torch
from .body import FlyBody, BodyParameters
from .gesture_batch import BatchGestureTrainer
from .gesture_training import GestureTrainer, Cancelled
from .gesture_scene import HandStimulus, GESTURES, random_placement, asset_digest
from .gesture_targets import muscle_reference, TARGET_VERSION, evaluate_reference, reference_geometry
from .gesture_gradients import muscle_activation
from .gesture_versions import descendants, stratified_gestures
from .policy_model import SensorActionTransformer, VERSION, eye_observation, observation_window, schema_digest
from .policy_simulation import PolicySimulation
from .policy_checkpoint import continuation_snapshot
from .policy_bc import RECIPE, run_behavior_cloning
from .senses import SensorSuite, SensorySettings


def policy_device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def policy_digest():
    digest = hashlib.sha256()
    for name in ('policy_model.py', 'policy_simulation.py', 'retina.py', 'senses.py', 'body.py',
                 'extended_mechanics.py', 'gesture_scene.py', 'embodied_vision.py', 'eye_surface.py'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def training_digest():
    """Training provenance is separate from inference compatibility for old weights."""
    digest = hashlib.sha256()
    for name in ('policy_training.py', 'policy_checkpoint.py', 'gesture_targets.py',
                 'gesture_gradients.py', 'gesture_versions.py', 'policy_bc.py', 'policy_evaluation.py',
                 'assets/gesture-target-routes.json'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def compatible_policy(digest, schema):
    current = policy_digest()
    if digest == current:
        return True
    # Only these audited, unchanged legacy inference paths are compatible.
    # New embodied optics have a distinct schema; never silently migrate weights.
    legacy = '6fd24934635849412c98f51027c809b26966ca2a6f593d65532f8131404bd889'
    grid = 'fb6b3a30889a696c59cd2e1222ce4bf718302d7768d665900810cd6a61533370'
    anatomical = 'fcc199d2fefdd54f521a74b17a1fbd52b4555789ef2cbf98b6505a2d0f48c322'
    extension = 'bdc012f40fba0ced3b17f295c6af57ede20af00c01cbb31ca585cc5dbdde3e56'
    shape = [(s.get('geometry'), s.get('samples')) for s in schema.get('sensors', [])]
    return current == '89d80a3c65bbf6b8d03a46b403b27d68226c3a7709c5260e31c5ed3727e68a44' and (
        digest == '73d86eee66311fa01a4ca44ffce3e558474e50812d2c0e378e09a9dfc6137899' and shape in [[('left', 857), ('right', 852)], [('balanced-left', 1024), ('balanced-right', 1024)], [('anatomical-left', 1024), ('anatomical-right', 1024)], [('embodied-left', 1024), ('embodied-right', 1024)], [('patch-left', 1024), ('patch-right', 1024)]] or
        digest == extension and shape in [[('left', 857), ('right', 852)], [('balanced-left', 1024), ('balanced-right', 1024)], [('anatomical-left', 1024), ('anatomical-right', 1024)], [('embodied-left', 1024), ('embodied-right', 1024)]] or
        digest in {legacy, grid, anatomical} and shape == [('left', 857), ('right', 852)] or
        digest in {grid, anatomical} and shape == [('balanced-left', 1024), ('balanced-right', 1024)] or
        digest == anatomical and shape == [('anatomical-left', 1024), ('anatomical-right', 1024)])



def policy_versions(directory):
    versions = []
    for path in Path(directory).glob('policy-*.json'):
        try:
            value = json.loads(path.read_text())
            if value['id'] != path.with_suffix('.pt').name or not path.with_suffix('.pt').exists():
                continue
            value['compatible'] = compatible_policy(value.get('policy_sha256'), value.get('schema', {}))
            versions.append(value)
        except (ValueError, KeyError, OSError):
            continue
    return sorted(versions, key=lambda v: v['created_at'])


def teacher_examples(settings, reference, cue, placement, history, chunk, stop):
    body = FlyBody(BodyParameters(**settings['body']), 'lab')
    suite = SensorSuite(SensorySettings(**settings['senses']))
    suite.visual_object = HandStimulus(cue, placement=placement)
    frames, examples = [], []
    for step, command in enumerate(reference.commands):
        if stop.is_set():
            raise Cancelled()
        frames.append(eye_observation(suite.sample(body, np.array([12., 3., .01]), 0.)))
        # Mask incomplete future chunks; do not invent repeated terminal labels.
        count = min(chunk, len(reference.commands) - step)
        target = np.zeros((chunk, body.model.nu), dtype=np.float32)
        target[:count] = reference.activations[step:step + count]
        examples.append((observation_window(frames, history), body.data.act.astype(np.float32).copy(), target, count))
        body.step_muscles(command, dt=.02)
    return examples


def example_loss(model, examples, physics_model, device):
    observations = {key: torch.tensor(np.stack([x[0][key] for x in examples]), device=device)
                    for key in examples[0][0]}
    commands = model(observations)
    state = torch.tensor(np.stack([x[1] for x in examples]), device=device)
    targets = torch.tensor(np.stack([x[2] for x in examples]), device=device)
    valid = torch.tensor([x[3] for x in examples], device=device)
    errors = []
    for step in range(model.chunk):
        action = torch.nn.functional.pad(commands[:, step].float(), (0, physics_model.nu - 84))
        state = muscle_activation(action, state, physics_model)
        errors.append((state[:, :84] - targets[:, step, :84]).square().mean(-1))
    mask = torch.arange(model.chunk, device=device)[None] < valid[:, None]
    # Each visual history has equal weight, even at the end of an episode.
    return (torch.stack(errors, 1).mul(mask).sum(1) / valid).mean()


class PolicyTrainer(BatchGestureTrainer):
    def __init__(self, directory):
        super().__init__(directory, Path('.'))

    def checkpoints(self):
        return [v['id'] for v in reversed(policy_versions(self.directory))]

    def snapshot(self, include_frame=True):
        value = GestureTrainer.snapshot(self, include_frame)
        return {**value, 'controller_kind': 'transformer', 'loaded_model': self.loaded_model,
                'versions': policy_versions(self.directory),
                'execution': {'device': policy_device(), 'precision': 'float32', 'gradient_precision': 'float32'},
                'algorithm': 'Behavior cloning · masked action L1 · shuffled minibatches · full-weight Adam'}

    def _load(self, name):
        if not name or Path(name).name != name or not name.startswith('policy-') or not name.endswith('.pt'):
            raise ValueError('Invalid policy checkpoint name')
        p = torch.load(self.directory / name, map_location='cpu', weights_only=True)
        if (p.get('version') != VERSION or not compatible_policy(p.get('policy_sha256'), p.get('schema', {}))
                or p.get('schema_sha256') != schema_digest(p['schema']) or p.get('assets_sha256') != asset_digest()):
            raise ValueError('Policy implementation, assets, or sensor/action schema differs')
        model = SensorActionTransformer.from_schema(p['schema'])
        if any(not torch.isfinite(v).all() for v in p['state_dict'].values()):
            raise ValueError('Nonfinite policy weights')
        model.load_state_dict(p['state_dict'], strict=True)
        return p

    def load_simulation(self, name):
        saved = self._load(name)
        model = SensorActionTransformer.from_schema(saved['schema'])
        model.load_state_dict(saved['state_dict'])
        result = PolicySimulation(model.to(policy_device()), saved['settings'])
        result.gesture_model = name
        return result

    def save_preview(self, name, epoch, sample, frame):
        folder = self.directory / 'previews' / Path(name).stem
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{epoch}-{sample}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(frame))
        temporary.replace(path)

    def load_preview(self, name, epoch, sample):
        if Path(name).name != name or not name.startswith('policy-') or not name.endswith('.pt'):
            raise ValueError('Invalid policy checkpoint name')
        if epoch < 1 or sample < 1 or not (self.directory / name).is_file():
            raise ValueError('Preview is unavailable')
        return json.loads((self.directory / 'previews' / Path(name).stem / f'{epoch}-{sample}.json').read_text())

    def delete_versions(self, version, expected_ids, loaded_model=None):
        with self.lock:
            if self.status['running'] or self.status.get('resumable'):
                raise ValueError('Finish or start a new run before deleting versions')
            affected = descendants(policy_versions(self.directory), version)
            if affected != sorted(expected_ids):
                raise ValueError('The branch changed; review deletion again')
            if loaded_model in affected:
                raise ValueError('Load another model before deleting the active model')
            for name in affected:
                (self.directory / name).unlink()
                (self.directory / name).with_suffix('.json').unlink(missing_ok=True)
                shutil.rmtree(self.directory / 'previews' / Path(name).stem, ignore_errors=True)
            if self.status.get('checkpoint') in affected:
                self.status['checkpoint'] = None
            return {'deleted': affected}

    def start(self, settings, *, iterations=100, horizon=25, batch_size=3, learning_rate=.0003,
              seed=42, checkpoint=None, proportions=None, chart_name='Visual policy', version_name='',
              early_stopping=True, early_stopping_patience=10, resume_from='latest',
              demonstrations_per_gesture=16, validation_episodes=4, rollout_episodes=4, evaluation_interval=10, **unused):
        proportions = {g: 1. for g in GESTURES} if proportions is None else proportions
        if (not set(proportions).issubset(GESTURES) or any(not np.isfinite(x) or x < 0 for x in proportions.values())
                or not np.isfinite(sum(proportions.values())) or sum(proportions.values()) <= 0):
            raise ValueError('Gesture proportions must be nonnegative with a positive total')
        if not (1 <= iterations <= 1000 and 5 <= horizon <= 250 and 1 <= batch_size <= 64 and 0 < learning_rate <= .1):
            raise ValueError('Invalid training limits')
        if any(type(value) is not int or not low <= value <= high for value, low, high in
               [(demonstrations_per_gesture, 1, 128), (validation_episodes, 1, 32),
                (rollout_episodes, 1, 32), (evaluation_interval, 1, 1000)]):
            raise ValueError('Invalid behavior-cloning dataset or evaluation settings')
        saved = self._load(checkpoint) if checkpoint else None
        if resume_from not in {'latest', 'best'}:
            raise ValueError('Resume point must be latest or best')
        continuation = continuation_snapshot(saved, resume_from)
        if self.status.get('resumable') and self.thread:
            self.stop_event.set()
            self.resume_event.set()
            self.thread.join()
        with self.lock:
            if self.status['running']:
                raise ValueError('Training is already running')
            settings = copy.deepcopy(saved['settings'] if saved else settings)
            from .policy_model import policy_vision_model
            vision_model = policy_vision_model(saved['schema']) if saved else 'compound-retina-balanced-v5'
            settings['senses'].update(vision_enabled=True, vision_model=vision_model, calibration_sphere=False)
            parent = saved['metadata'] if saved else None
            chart_id = parent['chart_id'] if parent else f'policy-chart-{time.time_ns()}'
            version = 1 + max((v['version'] for v in policy_versions(self.directory) if v['chart_id'] == chart_id), default=0)
            metadata = dict(id=f'policy-{time.time_ns()}.pt', chart_id=chart_id,
                chart_name=parent['chart_name'] if parent else chart_name.strip(), parent=checkpoint,
                version=version, name=version_name.strip() or f'v{version} · behavior cloning · {iterations} epochs',
                controller_kind='transformer', rank=0, iterations=iterations, horizon=horizon, batch_size=batch_size,
                learning_rate=learning_rate, seed=seed, created_at=datetime.now(timezone.utc).isoformat(),
                proportions={g: 100 * proportions.get(g, 0.) / sum(proportions.values()) for g in GESTURES},
                optimizer_start=(f'Restored Adam from {resume_from}; requested learning rate applied'
                    if continuation and parent.get('training_recipe') == RECIPE else
                    'Fresh Adam; changed to action-chunk behavior cloning' if continuation else
                    'Fresh Adam; legacy checkpoint has no optimizer history' if saved else 'Fresh Adam; random full weights'),
                training_recipe=RECIPE, loss_metric='action_mae',
                demonstrations_per_gesture=demonstrations_per_gesture, validation_episodes=validation_episodes,
                rollout_episodes=rollout_episodes, evaluation_interval=evaluation_interval,
                resume_from=resume_from if continuation else 'weights' if saved else None,
                parent_global_step=continuation['global_step'] if continuation else 0,
                parent_learning_rate=parent['learning_rate'] if parent else None,
                rng_start=('Restored training sample stream' if continuation and parent.get('training_recipe') == RECIPE and seed == parent['seed'] else
                           'Restarted sample stream with requested seed'),
                training_state_version=1,
                runtime_versions={'torch': str(torch.__version__), 'numpy': np.__version__},
                early_stopping=early_stopping, early_stopping_patience=early_stopping_patience,
                target_version=TARGET_VERSION, policy_sha256=policy_digest(), behavior_validated=False,
                execution={'device': policy_device(), 'precision': 'float32', 'gradient_precision': 'float32'}, batch_mode='parallel')
            self.status = dict(running=True, resumable=False, stopping=False, history=[], error=None, frame=None,
                phase='Preparing visual demonstrations', checkpoint=None, total=iterations, batch_size=batch_size,
                sample_total=horizon, iteration=0, wall_seconds=0.,
                optimizer_start=metadata['optimizer_start'], rng_start=metadata['rng_start'],
                training_recipe=RECIPE, loss_metric='action_mae',
                resume_from=metadata['resume_from'], parent_global_step=metadata['parent_global_step'])
            self.status['run_id'] = metadata['id']
            self.status['training_setup'] = {
                key: metadata[key] for key in ('iterations', 'batch_size', 'horizon', 'learning_rate',
                    'seed', 'early_stopping', 'early_stopping_patience', 'demonstrations_per_gesture',
                    'validation_episodes', 'rollout_episodes', 'evaluation_interval', 'chart_name')}
            self.status['training_setup'].update(checkpoint=checkpoint, rank=2,
                proportions=copy.deepcopy(proportions), resume_from=resume_from, version_name=version_name)
            self.stop_event.clear()
            self.pause_requested.clear()
            self.resume_event.clear()
            self.thread = threading.Thread(target=self._run_policy, args=(settings, saved, proportions, metadata), daemon=True)
            self.thread.start()

    def _run_policy(self, settings, saved, proportions, metadata):
        try:
            run_behavior_cloning(self, settings, saved, metadata)
        except Cancelled:
            self.publish(running=False, resumable=False, stopping=False, phase='Stopped · last completed epoch saved')
        except Exception as exc:
            self.publish(running=False, resumable=False, stopping=False, error=str(exc), phase='Training failed')

    def _save(self, model, state, settings, metadata, *, training_state=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / metadata['id']
        results = self.snapshot(include_frame=False)
        results.update(running=False, resumable=False, stopping=False, checkpoint=path.name, phase='Saved selected policy and latest training state')
        payload = {'version': VERSION, 'policy_sha256': policy_digest(), 'assets_sha256': asset_digest(),
                   'schema': model.schema(), 'schema_sha256': schema_digest(model.schema()),
                   'settings': settings, 'state_dict': state, 'metadata': metadata, 'results': results}
        if training_state is not None:
            payload['training_state'] = training_state
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        metadata['bytes'] = path.stat().st_size
        metadata['policy_sha256'] = payload['policy_sha256']
        index_temp = path.with_suffix('.json.tmp')
        index_temp.write_text(json.dumps(metadata, indent=2) + '\n')
        index_temp.replace(path.with_suffix('.json'))
        self.publish(checkpoint=path.name)
