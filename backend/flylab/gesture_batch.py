"""Online randomized batches, supervised muscle activation, adapter-only Adam.

GPU samples advance together with shared weights and independent neural/body state.
The CPU reference accumulates sample gradients sequentially.
BPTT spans the complete sample; 20 ms boundaries are recomputation checkpoints.
The last sample is published once after each completed iteration.
"""
import copy
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from .gesture_versions import version_metadata, descendants, stratified_gestures
import numpy as np
import torch

from .body import FlyBody, BodyParameters
from .neural import REGIONS
from .gesture_training import (GestureTrainer, GestureSession, Cancelled,
                               implementation_digest, inference_digest, compatible_inference, settings_from_simulation)
from .gesture_scene import GESTURES, HandStimulus, random_placement, asset_digest
from .gesture_gradients import GradientBrain, muscle_activation, DEFAULT_SURROGATE_SCALE
from .connectome_adapter import VERSION as ADAPTER_VERSION
from .paper_dynamics import PAPER_DYNAMICS_VERSION
from .gesture_optimization import adam_step, BestValidation, target_reachability
from .senses import SENSORY_VERSION

from .gesture_targets import muscle_reference, reference_geometry, evaluate_reference, TARGET_VERSION

VERSION = 'gesture-muscle-adam-v5'


def default_training_execution():
    gpu = torch.backends.mps.is_available()
    return {'device': 'mps' if gpu else 'cpu', 'precision': 'float16' if gpu else 'float64',
            'gradient_precision': 'float32' if gpu else 'float64'}


def target_muscle_indices(body, cue):
    """Supervise the moving leg and all stance muscles; never mask neuron updates."""
    if cue not in GESTURES:
        raise ValueError('Unknown hand gesture')
    return np.arange(84, dtype=np.int64)


def demonstrations(session, horizon, stop, cues=None, available_channels=None):
    """Targets from independently replayed 50 Hz commands, absent from inference."""
    parameters = BodyParameters(**session.settings['body'])
    if available_channels is None and hasattr(session, 'sim'):
        available_channels = tuple(i for i, ids in enumerate(session.sim.bridge.channels[:84]) if len(ids))
    targets, evidence = {}, {}
    for cue in GESTURES if cues is None else cues:
        reference = muscle_reference(cue, horizon, parameters, stop, available_channels)
        targets[cue] = torch.tensor(reference.activations, dtype=torch.float64)
        evidence[cue] = reference.evidence
        if not reference.evidence['pose_reached']:
            raise ValueError(f'{cue}: reference did not hold the pose at these body settings and duration; no training started')
    return targets, evidence


class BatchSession(GestureSession):
    def __init__(self, graph, settings, rank, seed):
        super().__init__(graph, settings, rank, seed)
        self.settings = settings
        self.execution_lock = threading.RLock()
        if self.sim.full_brain.device.type == 'mps':
            from .gesture_metal import MetalGradientBrain
            self.gradient = MetalGradientBrain(self.sim.full_brain, self.adapter, self.sim.bridge,
                surrogate_scale=settings.get('surrogate_scale', DEFAULT_SURROGATE_SCALE))
        else:
            self.gradient = GradientBrain(self.sim.full_brain, self.adapter, self.sim.bridge,
                surrogate_scale=settings.get('surrogate_scale', DEFAULT_SURROGATE_SCALE))
        self.gradient.full_history = True
        self.gradient.stable_history = True

    def parallel_samples(self, cues, placements, targets, stop, progress=None):
        """One shared GPU brain graph; separate sensory/body and neural state per lane."""
        from .gesture_parallel import ParallelMetalBrain
        from .senses import SensorSuite, SensorySettings
        sim, engine = self.sim, self.gradient
        count = len(cues)
        if count != len(placements) or not count:
            raise ValueError('Each sample requires a gesture and placement')
        with self.execution_lock:
            sim.reset()
            sim.full_brain.generator.manual_seed(self.seed)
            engine.reset()
            engine.sync_weights()
            if getattr(self, '_parallel_size', None) != count:
                self._parallel = ParallelMetalBrain(engine, count)
                self._parallel_bodies = [FlyBody(BodyParameters(**self.settings['body']), sim.body.scene_id)
                                         for _ in range(count - 1)] + [sim.body]
                self._parallel_size = count
            else:
                self._parallel.reset()
            parallel, bodies = self._parallel, self._parallel_bodies
            sensors = [SensorSuite(SensorySettings(**self.settings['senses'])) for _ in cues]
            for body, suite, cue, placement in zip(bodies, sensors, cues, placements):
                body.reset()
                suite.visual_object = HandStimulus(str(cue), placement=placement)
            target = torch.stack([targets[str(cue)] for cue in cues]).to('mps', dtype=torch.float32)
            horizon = target.shape[1]
            selected = [target_muscle_indices(body, str(cue)) for body, cue in zip(bodies, cues)]
            weights = np.zeros((count, sim.body.model.nu), dtype=np.float32)
            for i, ids in enumerate(selected):
                weights[i, ids] = 1 / len(ids)
            weights = torch.tensor(weights, device='mps')
            from .gesture_history import check_history_memory
            check_history_memory(parallel, horizon)
            sample_loss = None
            previous_activation = None
            totals = np.zeros(count)
            began = time.perf_counter()
        for step in range(horizon):
            if stop.is_set():
                raise Cancelled()
            with self.execution_lock:
                # MuJoCo and ray sensing remain on CPU. Neural kernels span all
                # samples in one dispatch; no graph or adapter copy per sample.
                commands = [sim.bridge.prepare_command(body, sim.source, 0.,
                    sim.environment_mode == 'spatial', frame=suite.sample(body, sim.source, 0.,
                        sim.environment_mode == 'spatial', duration_s=.02))
                    for body, suite in zip(bodies, sensors)]
                parallel.advance(torch.stack([x[0] for x in commands]),
                                 silence=torch.stack([x[1] for x in commands]))
                action = parallel.muscles()
                previous = (previous_activation if previous_activation is not None else
                            torch.tensor(np.stack([body.data.act.copy() for body in bodies]), device='mps', dtype=torch.float32))
                activation = muscle_activation(action, previous, sim.body.model)
                losses = ((activation - target[:, step]).square() * weights).sum(dim=1)
                sample_loss = losses.mean() / horizon if sample_loss is None else sample_loss + losses.mean() / horizon
                previous_activation = activation
                actions, activations = action.detach().cpu().numpy(), activation.detach().cpu().numpy()
                for body, command, predicted in zip(bodies, actions, activations):
                    body.step_muscles(command, dt=.02)
                    if not np.allclose(predicted, body.data.act, atol=2e-6, rtol=2e-5):
                        raise ValueError('Differentiable activation differs from MuJoCo')
                totals += losses.detach().cpu().numpy() / horizon

            if progress:
                progress(sample_step=step + 1)
        if stop.is_set():
            raise Cancelled()
        with self.execution_lock:
            parallel.export_sample(count - 1)
            sim.sensors = sensors[-1]
            sim.steps, sim.time = horizon, horizon / 50
            sim.wall_seconds = time.perf_counter() - began
            for region in REGIONS:
                ids = sim.bridge.regions[region['id']]
                sim.state[region['start']:region['end']] = float(sim.full_brain.rates[ids].mean() / 100) if len(ids) else 0.
            frame = sim.snapshot()
            frame['gesture_stimulus'] = sim.sensors.visual_object.summary()
            frame['model']['ready'] = True
            samples = [{'loss': float(loss), 'gesture': str(cue), 'placement': placement,
                        'pose': evaluate_reference(body, str(cue), reference_geometry(body.parameters)[1]),
                        'final_joints': body.data.qpos[body.qadr].tolist()}
                       for loss, cue, placement, body in zip(totals, cues, placements, bodies)]
            ids = selected[-1]
            comparison = {'names': [sim.body.model.actuator(int(i)).name for i in ids],
                          'target': target[-1, -1].cpu().numpy()[ids].tolist(),
                          'actual': sim.body.data.act[ids].tolist()}
        if progress:
            progress(frame=frame, muscle_comparison=comparison, last_batch=samples,
                     preview_gesture=str(cues[-1]), loss=float(np.mean(totals)),
                     phase='Backpropagating full sample', backward_step=0, backward_total=horizon)
        with self.execution_lock:
            from .gesture_history import backward_full_history
            backward_full_history(parallel, sample_loss, stop, progress=progress)
            parallel.detach_state()
        return samples, frame, comparison

    def sample(self, cue, placement, target, stop, scale, train=True, progress=None):
        sim, engine = self.sim, self.gradient
        with self.execution_lock:
            sim.reset()
            sim.full_brain.generator.manual_seed(self.seed)
            engine.reset()
            engine.sync_weights()
            # Reset initializes baseline regional output gains, but does not alter W.
            sim.sensors.visual_object = HandStimulus(cue, placement=placement)
            selected = target_muscle_indices(sim.body, cue)
            gpu = engine.parameters.device.type == 'mps'
            target = target.to(engine.parameters.device, dtype=engine.parameters.dtype)
            selected_tensor = torch.tensor(selected, device=engine.parameters.device)
            total_loss = 0.
            sample_loss = None
            previous_activation = None
            if gpu and train:
                from .gesture_history import check_history_memory
                check_history_memory(engine, len(target))
            began = time.perf_counter()
        for step in range(len(target)):
            if stop.is_set():
                raise Cancelled()
            with self.execution_lock:
                with torch.set_grad_enabled(train):
                    drive, mask = sim.bridge.prepare_command(sim.body, sim.source, 0.,
                        sim.environment_mode == 'spatial', frame=sim.sensory_frame())
                    engine.advance(drive, silence=mask)
                    action = engine.muscles()
                    previous = (previous_activation if previous_activation is not None else
                                torch.tensor(sim.body.data.act.copy(), device=action.device, dtype=action.dtype))
                    activation = muscle_activation(action, previous, sim.body.model)
                    loss = (activation[selected_tensor] - target[step, selected_tensor]).square().mean()
                    if train:
                        sample_loss = loss * scale / len(target) if sample_loss is None else sample_loss + loss * scale / len(target)
                    previous_activation = activation
                sim.body.step_muscles(action.detach().cpu().numpy(), dt=.02)
                # An implementation mismatch must fail instead of reporting a proxy
                # as the body's activation loss.
                if not np.allclose(activation.detach().cpu().numpy(), sim.body.data.act,
                                   atol=2e-6 if gpu else 1e-8, rtol=2e-5 if gpu else 1e-7):
                    raise ValueError('Differentiable activation differs from MuJoCo')
                total_loss += float(loss.detach()) / len(target)

                sim.steps += 1
                sim.time = sim.steps / 50
            if progress:
                progress(sample_step=step + 1)
        with self.execution_lock:
            if train:
                if gpu:
                    from .gesture_history import backward_full_history
                    backward_full_history(engine, sample_loss, stop)
                else:
                    sample_loss.backward()
            engine.detach_state()
            sim.wall_seconds = time.perf_counter() - began
            for region in REGIONS:
                ids = sim.bridge.regions[region['id']]
                sim.state[region['start']:region['end']] = float(sim.full_brain.rates[ids].mean() / 100) if len(ids) else 0.
            frame = sim.snapshot()
            frame['gesture_stimulus'] = sim.sensors.visual_object.summary()
            frame['model']['ready'] = True
            names = [sim.body.model.actuator(int(i)).name for i in selected]
            comparison = {'names': names, 'target': target[-1, selected_tensor].tolist(),
                          'actual': sim.body.data.act[selected].tolist()}
        return {'loss': total_loss, 'gesture': cue, 'placement': placement,
                'pose': evaluate_reference(sim.body, cue, reference_geometry(sim.body.parameters)[1]),
                'final_joints': sim.body.data.qpos[sim.body.qadr].tolist()}, frame, comparison


class BatchGestureTrainer(GestureTrainer):
    def __init__(self, directory, graph, session_factory=BatchSession, reference_factory=demonstrations):
        super().__init__(directory, graph, session_factory)
        self.reference_factory = reference_factory
        self.loaded_model = None
        # API injects the live workbench lock: Metal submissions and device
        # synchronization must never overlap across training and inference.
        self.execution_lock = threading.RLock()
        self.pause_requested = threading.Event()
        self.resume_event = threading.Event()

    def request_stop(self):
        with self.lock:
            if self.status['running']:
                self.pause_requested.set()
                self.publish(stopping=True)
        return self.snapshot()

    def resume(self):
        with self.lock:
            if not self.status.get('resumable') or not self.thread or not self.thread.is_alive():
                raise ValueError('No paused training run to resume')
            self.pause_requested.clear()
            self.publish(running=True, resumable=False, stopping=False, phase='Resuming training')
            self.resume_event.set()
        return self.snapshot()

    def _pause_boundary(self):
        if not self.pause_requested.is_set():
            return 0.
        began = time.perf_counter()
        self.resume_event.clear()
        self.publish(running=False, resumable=True, stopping=False,
                     phase='Stopped · ready to resume')
        while not self.resume_event.wait(.2):
            if self.stop_event.is_set():
                raise Cancelled()
        if self.stop_event.is_set():
            raise Cancelled()
        return time.perf_counter() - began

    def checkpoints(self):
        return [x['id'] for x in reversed(version_metadata(self.directory))]

    def delete_versions(self, version, expected_ids, loaded_model=None):
        with self.lock:
            if self.status['running'] or self.status.get('resumable'):
                raise ValueError('Finish or discard training before deleting weight versions')
            affected = descendants(version_metadata(self.directory), version)
            if sorted(expected_ids) != affected:
                raise ValueError('The branch changed. Review the deletion again.')
            if loaded_model in affected:
                raise ValueError('Load the base or another branch before deleting the active model')
            for name in affected:
                (self.directory / name).unlink()
                (self.directory / name).with_suffix('.json').unlink(missing_ok=True)
            if self.status.get('checkpoint') in affected:
                self.status['checkpoint'] = None
            return {'deleted': affected}

    def snapshot(self, include_frame=True):
        return {**super().snapshot(include_frame), 'loaded_model': self.loaded_model,
                'execution': self.status.get('execution', default_training_execution()),
                **({'phase': 'Stopping · finishing current batch'} if self.status.get('stopping') else {}),
                'algorithm': 'Supervised muscle activation · Adam · surrogate gradients',
                'surrogate_scale_default': DEFAULT_SURROGATE_SCALE,
                'backprop_window_ms': self.status.get('backprop_window_ms', 20), 'versions': version_metadata(self.directory)}

    def _load(self, name):
        if not name or Path(name).name != name or not name.startswith('gesture-') or not name.endswith('.pt'):
            raise ValueError('Invalid gesture checkpoint name')
        p = torch.load(self.directory / name, map_location='cpu', weights_only=True)
        graph = json.loads((self.graph / 'manifest.json').read_text())
        if p.get('version') not in {VERSION, 'gesture-muscle-adam-v4'} or not compatible_inference(p):
            raise ValueError('Incompatible gesture model: forward implementation or schema')
        for key, value in {'graph_sha256': graph['graph_sha256'],
                           'adapter_version': ADAPTER_VERSION, 'assets_sha256': asset_digest(),
                           'dynamics_version': PAPER_DYNAMICS_VERSION, 'sensory_version': SENSORY_VERSION}.items():
            if p.get(key) != value:
                raise ValueError(f'Incompatible gesture model: {key}')
        rank, params = p['rank'], p['parameters']
        if not isinstance(rank, int) or not 1 <= rank <= 64 or not isinstance(params, torch.Tensor) or params.shape != (2, len(p['groups']), rank) or not torch.isfinite(params).all() or params.abs().max() > 4:
            raise ValueError('Invalid model adapter')
        return p

    def start(self, settings, *, iterations=10, horizon=25, rank=2, seed=42,
              batch_size=3, learning_rate=.01, gesture=None, checkpoint=None,
              proportions=None, chart_name="Hand gestures", version_name="",
              early_stopping=True, early_stopping_patience=10, surrogate_scale=DEFAULT_SURROGATE_SCALE):
        if not isinstance(early_stopping, bool) or type(early_stopping_patience) is not int or not 1 <= early_stopping_patience <= 1000:
            raise ValueError('Invalid early stopping settings')
        if not isinstance(surrogate_scale, (int, float)) or isinstance(surrogate_scale, bool) or not np.isfinite(surrogate_scale) or not 0 < surrogate_scale <= 1:
            raise ValueError('Surrogate scale must be in (0, 1]')
        if gesture is not None:
            raise ValueError('Select a hand in the live 3D scene for inference')
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (iterations, horizon, rank, seed, batch_size)) or not (1 <= iterations <= 1000 and 5 <= horizon <= 250 and 1 <= rank <= 64 and 0 <= seed <= 2147483647 and 1 <= batch_size <= 64 and np.isfinite(learning_rate) and 0 < learning_rate <= .1):
            raise ValueError('Invalid batch training settings')
        proportions = {g: 1. for g in GESTURES} if proportions is None else proportions
        if not set(proportions).issubset(GESTURES) or any(not np.isfinite(x) or x < 0 for x in proportions.values()) or not np.isfinite(sum(proportions.values())) or sum(proportions.values()) <= 0:
            raise ValueError('Specify nonnegative gesture proportions with a positive total')
        proportions = {g: proportions.get(g, 0.) for g in GESTURES}
        if not isinstance(chart_name, str) or not 1 <= len(chart_name.strip()) <= 80 or not isinstance(version_name, str) or len(version_name) > 120:
            raise ValueError('Invalid chart or version name')
        saved = self._load(checkpoint) if checkpoint else None
        if saved:
            settings, rank = saved['settings'], saved['rank']
        settings = copy.deepcopy(settings)
        settings.setdefault('training_execution', default_training_execution())
        settings['adapter_grouping'] = 'motor-target-v2'
        settings['surrogate_scale'] = float(surrogate_scale)
        execution = {**settings['training_execution'],
            'gradient_precision': 'float32' if settings['training_execution']['device'] == 'mps' else 'float64'}
        parent_meta = saved['metadata'] if saved else None
        chart_id = parent_meta['chart_id'] if parent_meta else f'chart-{time.time_ns()}'
        versions = version_metadata(self.directory)
        version_number = 1 + max((x['version'] for x in versions if x['chart_id'] == chart_id), default=0)
        percentages = {g: round(100 * x / sum(proportions.values()), 1) for g, x in proportions.items()}
        metadata = {'adapter_grouping': settings['adapter_grouping'], 'target_version': TARGET_VERSION, 'chart_id': chart_id, 'chart_name': parent_meta['chart_name'] if parent_meta else chart_name.strip(),
            'parent': checkpoint, 'version': version_number,
            'compatible': True, 'name': version_name.strip() or f"v{version_number} · " + " / ".join(f"{GESTURES[g]} {x:g}%" for g, x in percentages.items() if x) + f" · rank {rank}",
            'proportions': percentages, 'iterations': iterations, 'batch_size': batch_size, 'rank': rank,
            'horizon': horizon, 'backprop_window_ms': horizon * 20,
            'gradient_history': 'full-sample-checkpointed',
            'learning_rate': learning_rate, 'seed': seed, 'execution': execution,
            'batch_mode': 'parallel' if execution['device'] == 'mps' else 'sequential',
            'surrogate_scale': float(surrogate_scale),
            'early_stopping': early_stopping, 'early_stopping_patience': early_stopping_patience,
            'optimizer_start': 'fresh Adam; inherited adapter' if saved else 'fresh Adam; identity adapter'}
        if not (self.graph / 'manifest.json').exists():
            raise ValueError('Full MaleCNS graph is required')
        # A new run explicitly discards a paused continuation before replacing status.
        if self.status.get('resumable') and self.thread:
            self.stop_event.set()
            self.resume_event.set()
            self.thread.join()
        with self.lock:
            if self.status['running']:
                raise ValueError('Training is already running')
            self.stop_event.clear()
            self.pause_requested.clear()
            self.resume_event.clear()
            self.status = {'running': True, 'mode': 'training', 'phase': 'Loading full brain',
                'started_at': time.time(), 'iteration': 0, 'total': iterations, 'batch_size': batch_size, 'batch_sample': 0,
                'history': [], 'frame': None, 'preview_iteration': 0, 'error': None,
                'checkpoint': None, 'rank': rank, 'horizon': horizon, 'learning_rate': learning_rate,
                'surrogate_scale': float(surrogate_scale),
                'backprop_window_ms': horizon * 20, 'gradient_history': 'full-sample-checkpointed',
                'seed': seed, 'cancelled': False, 'completed_rollouts': 0, 'parent': checkpoint,
                'execution': execution}
            self.status['run_id'] = f'gesture-run-{time.time_ns()}'
            self.status['training_setup'] = {
                key: metadata[key] for key in ('iterations', 'batch_size', 'horizon', 'rank', 'learning_rate',
                    'seed', 'early_stopping', 'early_stopping_patience', 'chart_name')}
            self.status['training_setup'].update(checkpoint=checkpoint, proportions=copy.deepcopy(proportions),
                version_name=version_name, resume_from='latest', demonstrations_per_gesture=16,
                validation_episodes=4, rollout_episodes=4, evaluation_interval=10)
            self.status['parallel_samples'] = batch_size if execution['device'] == 'mps' else 1
            self.thread = threading.Thread(target=self._batch_run,
                args=(copy.deepcopy(settings), iterations, horizon, rank, seed, batch_size, learning_rate, proportions, metadata, saved), daemon=True)
            self.thread.start()

    def _batch_run(self, settings, iterations, horizon, rank, seed, batch_size, learning_rate, proportions, metadata, saved):
        began, temporary = time.perf_counter(), None
        try:
            with self.execution_lock:
                session = self.session_factory(self.graph, settings, rank, seed)
                session.execution_lock = self.execution_lock
                engine = session.gradient
                if saved:
                    with torch.no_grad():
                        engine.parameters.copy_(session.adapter.inherit(saved['groups'], saved['parameters']))
            self.publish(phase='Generating muscle demonstrations', adapter=session.adapter.summary())
            targets, evidence = self.reference_factory(session, horizon, self.stop_event,
                cues=[cue for cue, amount in proportions.items() if amount > 0])
            self.publish(teacher_evaluation=evidence, target_reachability=target_reachability(session, targets, proportions))
            optimizer = torch.optim.Adam([engine.parameters], lr=learning_rate)
            rng = np.random.default_rng(seed)
            history = []
            # One fixed placement per selected class; weights match the configured
            # mix exactly even when a small randomized batch omits a class.
            validation_rng = np.random.default_rng(seed + 1)
            validation_inputs = [(cue, random_placement(validation_rng))
                                 for cue, amount in proportions.items() if amount > 0]
            def evaluate():
                result = 0.
                for cue, placement in validation_inputs:
                    if self.stop_event.is_set():
                        raise Cancelled()
                    measured, _, _ = session.sample(cue, placement, targets[cue],
                                                   self.stop_event, 0., train=False)
                    result += measured['loss'] * proportions[cue] / sum(proportions.values())
                return result
            self.publish(phase='Measuring initial validation loss')
            validation_loss = evaluate()
            initial_validation_loss = validation_loss
            with self.execution_lock:
                best = BestValidation(engine.parameters, validation_loss, metadata["early_stopping_patience"])
            metadata.update(optimizer_policy="Adam with best checkpoint", early_stopping_patience=best.patience)
            self.publish(validation_initial_loss=validation_loss, validation_loss=validation_loss,
                         validation_inputs=[{'gesture': cue, 'placement': placement}
                                            for cue, placement in validation_inputs])
            for iteration in range(1, iterations + 1):
                began += self._pause_boundary()
                with self.execution_lock:
                    optimizer.zero_grad(set_to_none=True)
                    engine.sync_weights()
                losses, samples = [], []
                training_started = time.perf_counter()
                # Stratified by the configured gesture mix; randomized order and every
                # transform are generated on demand, not from a cached dataset.
                cues = stratified_gestures(rng, proportions, batch_size)
                if engine.parameters.device.type == 'mps':
                    placements = [random_placement(rng) for _ in cues]
                    self.publish(phase='Training parallel batch', iteration=iteration,
                        batch_sample=batch_size, sample_step=0, sample_total=horizon,
                        current_gesture=str(cues[-1]), batch_gestures=[str(cue) for cue in cues],
                        backward_step=0, backward_total=horizon)
                    samples, frame, comparison = session.parallel_samples(cues, placements, targets,
                        self.stop_event, progress=lambda **values: self.publish(
                            **({'preview_iteration': iteration} if 'frame' in values else {}),
                            wall_seconds=time.perf_counter() - began, **values))
                    losses = [sample['loss'] for sample in samples]
                    self.publish(completed_rollouts=iteration * batch_size,
                                 wall_seconds=time.perf_counter() - began)
                for index, cue in enumerate(cues if engine.parameters.device.type != 'mps' else []):
                    if self.stop_event.is_set():
                        raise Cancelled()
                    self.publish(phase='Training batch', iteration=iteration, batch_sample=index + 1,
                                 sample_step=0, sample_total=horizon, current_gesture=str(cue))
                    placement = random_placement(rng)
                    result, frame, comparison = session.sample(str(cue), placement, targets[str(cue)],
                        self.stop_event, 1 / batch_size, progress=lambda **values: self.publish(
                            wall_seconds=time.perf_counter() - began, **values))
                    losses.append(result['loss'])
                    samples.append(result)
                    self.publish(completed_rollouts=(iteration - 1) * batch_size + index + 1,
                                 wall_seconds=time.perf_counter() - began)
                if self.stop_event.is_set():
                    raise Cancelled()
                with self.execution_lock:
                    from .gesture_history import clip_full_gradient
                    norm, log_norm = clip_full_gradient(engine)
                    training_seconds = time.perf_counter() - training_started
                    validation_started = time.perf_counter()
                    self.publish(phase='Validating adapter update')
                    validation = adam_step(optimizer, engine.parameters, evaluate, validation_loss)
                    validation_loss = validation['validation_loss']
                loss = float(np.mean(losses))
                history.append({'iteration': iteration, 'loss': loss,
                                'gradient_norm': norm, 'gradient_log10_norm': log_norm, 'sample_losses': losses,
                                'training_seconds': training_seconds,
                                'validation_seconds': time.perf_counter() - validation_started, **validation})
                # GPU scenes were published before backward; do not publish twice.
                self.publish(history=history.copy(),
                    **({'frame': frame} if engine.parameters.device.type != 'mps' else {}),
                    muscle_comparison=comparison,
                    preview_iteration=iteration, preview_gesture=samples[-1]['gesture'],
                    last_batch=samples, loss=loss, gradient_norm=norm, gradient_log10_norm=log_norm, **validation,
                    wall_seconds=time.perf_counter() - began)
                with self.execution_lock:
                    should_stop = best.observe(engine.parameters, validation_loss, iteration)
                self.publish(best_validation_loss=best.loss, best_iteration=best.iteration,
                             early_stopping_patience=best.patience,
                             iterations_without_improvement=best.stalled)
                if should_stop and metadata["early_stopping"]:
                    self.publish(early_stopped=True, stop_reason=f'No meaningful validation improvement for {best.patience} iterations')
                    break
            # Training continues from the latest Adam state; only the export uses
            # the best measured adapter, which may be the starting weights.
            with self.execution_lock:
                with torch.no_grad():
                    engine.parameters.copy_(best.parameters)
                engine.sync_weights()
                self.publish(adapter=session.adapter.summary(), phase=(
                    'Stopped improving; best validated weights saved' if best.stalled >= best.patience and metadata['early_stopping']
                    else 'Training complete; best validated weights saved') if best.iteration else
                    'Complete; starting weights remained best', success=None, behavior_validated=False)
                name = f'gesture-{time.time_ns()}.pt'
                metadata.update(id=name, implementation_sha256=implementation_digest(), inference_sha256=inference_digest(), created_at=datetime.now(timezone.utc).isoformat(),
                                iterations_completed=len(history), final_loss=history[-1]['loss'], validation_initial_loss=initial_validation_loss,
                                validation_loss=best.loss, best_iteration=best.iteration, last_validation_loss=validation_loss, accepted_updates=sum(row['update_accepted'] for row in history),
                                target_reachability=target_reachability(session, targets, proportions), wall_seconds=time.perf_counter() - began,
                                trainable_parameters=engine.parameters.numel(), teacher_evaluation=evidence,
                                behavior_validated=False)
                payload = {'target_version': TARGET_VERSION, 'version': VERSION, 'adapter_version': ADAPTER_VERSION,
                    'dynamics_version': PAPER_DYNAMICS_VERSION, 'sensory_version': SENSORY_VERSION,
                    'graph_sha256': session.sim.full_brain.manifest['graph_sha256'],
                    'assets_sha256': asset_digest(), 'implementation_sha256': implementation_digest(), 'inference_sha256': inference_digest(),
                    'settings': settings, 'rank': rank, 'seed': seed, 'batch_size': batch_size,
                    'learning_rate': learning_rate, 'horizon': horizon, 'backprop_window_ms': horizon * 20, 'gradient_history': 'full-sample-checkpointed',
                    'groups': [list(g) for g in session.adapter.groups],
                    'parameters': engine.parameters.detach().cpu().clone(),
                    'metadata': metadata, 'results': self.snapshot(False)}
                payload['results'].update(running=False, checkpoint=name)
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.directory / (name + '.tmp')
            torch.save(payload, temporary)
            if self.stop_event.is_set():
                raise Cancelled()
            temporary.replace(self.directory / name)
            temporary = None
            metadata['bytes'] = (self.directory / name).stat().st_size
            meta_path = (self.directory / name).with_suffix('.json')
            meta_path.with_suffix('.json.tmp').write_text(json.dumps(metadata, indent=2))
            meta_path.with_suffix('.json.tmp').replace(meta_path)
            self.publish(checkpoint=name)
        except Cancelled:
            self.publish(cancelled=True, phase='Stopped; no new model saved')
        except Exception as exc:
            self.publish(error=str(exc), phase='Training failed')
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
            self.publish(running=False, resumable=False, stopping=False, wall_seconds=time.perf_counter() - began)
