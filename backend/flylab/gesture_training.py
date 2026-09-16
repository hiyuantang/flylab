"""Supervised pose matching through an isolated, complete connectome simulation.

SPSA estimates changes in pose loss by actually running the spiking brain and
MuJoCo. No surrogate brain, gesture classifier, pose tracker or direct joint
assignment participates in inference. Geometric target poses only score runs.
"""
from dataclasses import asdict, replace
from pathlib import Path
import copy
import hashlib
import json
import threading
import time

import mujoco
import numpy as np
import torch

from .body import BodyParameters, LEGS
from .connectome_adapter import ConnectomeAdapter, VERSION as ADAPTER_VERSION
from .full_connectome import Physiology
from .gesture_scene import HandStimulus, GESTURES, RAISED_FRONT_LEGS, catalogue, asset_digest
from .motor_mapping import MAPPING_PROFILE
from .paper_dynamics import PAPER_DYNAMICS_VERSION
from .senses import SensorSuite, SensorySettings, SENSORY_VERSION
from .simulation import Simulation

VERSION = 'gesture-pose-training-v1'


def implementation_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name in {'api.py', 'gesture_versions.py'}:
            continue  # UI routing and version bookkeeping do not change model execution.
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def inference_digest():
    """Forward-model compatibility is independent of training-only algorithms."""
    excluded = {'api.py', 'gesture_versions.py', 'gesture_batch.py', 'gesture_gradients.py',
                'gesture_metal.py', 'gesture_parallel.py', 'gesture_targets.py',
                'gesture_training.py', 'gesture_optimization.py', 'gesture_history.py'}
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name not in excluded and not path.name.startswith('policy_'):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def compatible_inference(payload):
    current = inference_digest()
    if payload.get('inference_sha256') is not None:
        return payload['inference_sha256'] == current
    # Audited pre-optimization v4: identical forward dynamics, topology, adapter,
    # sensory routing and motor mapping. Never allow this across forward changes.
    audited = {'d369fb4475d3a95e13337bb1a2aebf9011a1020435f2860804750ae7b65d5fb6':
               '57f7d3490183f2ec42ee48e39f4ae1b9f3b3835577b9581bbc3207394637d9ba'}
    return (payload.get('implementation_sha256') == implementation_digest()
            or audited.get(payload.get('implementation_sha256')) == current)


class Cancelled(Exception):
    pass


def target_poses(body):
    """Geometric demonstrations via constrained foot IK, never an action path."""
    targets = {}
    rest = body.data.qpos.copy()
    feet = body.data.site_xpos[body.foot_ids].copy()
    limits = body.model.jnt_range[body.active_ids]
    for gesture in GESTURES:
        data = mujoco.MjData(body.model)
        data.qpos[:] = rest
        desired = feet.copy()
        if gesture == 'fist':
            data.qpos[2] -= .18
        for leg in RAISED_FRONT_LEGS.get(gesture, ()):
            desired[LEGS.index(leg), 2] += .35
        for _ in range(60):
            mujoco.mj_forward(body.model, data)
            errors = desired - data.site_xpos[body.foot_ids]
            if np.max(np.linalg.norm(errors, axis=1)) < .005:
                break
            for leg, site in enumerate(body.foot_ids):
                columns = slice(leg * 7, (leg + 1) * 7)
                jac = np.zeros((3, body.model.nv))
                mujoco.mj_jacSite(body.model, data, jac, None, site)
                local = jac[:, body.vadr[columns]]
                change = local.T @ np.linalg.solve(local @ local.T + np.eye(3) * .01, errors[leg])
                q = body.qadr[columns]
                data.qpos[q] = np.clip(data.qpos[q] + np.clip(change, -.08, .08),
                                      limits[columns, 0], limits[columns, 1])
        mujoco.mj_forward(body.model, data)
        residual = float(np.max(np.linalg.norm(desired - data.site_xpos[body.foot_ids], axis=1)))
        if residual > .05:
            raise ValueError(f'Cannot construct {gesture} target: foot IK residual {residual:.3f} mm')
        targets[gesture] = {'joints': data.qpos[body.qadr].copy(), 'height': float(data.qpos[2]),
                            'feet': data.site_xpos[body.foot_ids].copy(), 'ik_residual_mm': residual}
    return targets


def pose_error(body, target):
    joint = float(np.sqrt(np.mean((body.data.qpos[body.qadr] - target['joints']) ** 2)))
    height = float(abs(body.data.qpos[2] - target['height']))
    rotation = body.data.xmat[body.body_ids['c_thorax']].reshape(3, 3)
    upright = float(rotation[2, 2])
    foot_errors = np.linalg.norm(body.data.site_xpos[body.foot_ids] - target['feet'], axis=1)
    foot = float(foot_errors.max())
    loss = (joint / .2) ** 2 + (height / .2) ** 2 + float(np.mean((foot_errors / .2) ** 2)) + 4 * (1 - upright) ** 2
    if not np.isfinite(loss):
        raise ValueError('Nonfinite pose loss')
    return {'loss': loss, 'joint_rmse_rad': joint, 'height_error_mm': height, 'upright': upright,
            'max_foot_error_mm': foot,
            'pose_reached': bool(joint < .1 and height < .15 and foot < .12 and upright > .9)}


class GestureSession:
    def __init__(self, graph, settings, rank, seed):
        self.sim = Simulation()
        execution = settings.get('training_execution', {'device': 'cpu', 'precision': 'float64'})
        self.sim.configure('connectome', graph, physiology=Physiology.paper(),
                           body_parameters=BodyParameters(**settings['body']),
                           mapping_profile=settings['mapping_profile'], device=execution['device'], precision=execution['precision'],
                           command_hz=50, coupling_mode='serial')
        self.sim.bridge.set_parameters(np.array(settings['gains']))
        self.sim.sensors = SensorSuite(SensorySettings(**settings['senses']))
        self.sim.odor, self.sim.intensity = 'none', 0.
        self.adapter = ConnectomeAdapter(self.sim.full_brain, rank, seed,
                                         grouping=settings.get('adapter_grouping', 'superclass-side-v1'))
        self.targets = target_poses(self.sim.body)
        self.seed = seed

    def rollout(self, parameters, gesture, horizon, variant, stop, publish, *, blank=False):
        self.sim.reset()
        self.sim.full_brain.generator.manual_seed(self.seed)
        self.adapter.apply(parameters)
        self.sim.sensors.visual_object = None if blank else HandStimulus(gesture, variant=variant)
        errors = []
        for step in range(horizon):
            if stop.is_set():
                raise Cancelled()
            self.sim.advance()
            measured = pose_error(self.sim.body, self.targets[gesture])
            if step >= horizon // 2:
                errors.append(measured['loss'])
            if step == 0 or step == horizon - 1 or step % 5 == 0:
                frame = self.sim.snapshot()
                frame['gesture_stimulus'] = self.sim.sensors.visual_object.summary() if self.sim.sensors.visual_object else None
                frame['model']['ready'] = True
                target = self.targets[gesture]
                publish(frame=frame, current_gesture=gesture, sample_step=step + 1,
                        sample_total=horizon, current_error=measured,
                        pose_comparison={'joint_names': [self.sim.body.model.joint(int(i)).name for i in self.sim.body.active_ids],
                                         'target': target['joints'].tolist(),
                                         'actual': self.sim.body.data.qpos[self.sim.body.qadr].tolist()},
                        adapter=self.adapter.summary())
        return {**measured, 'loss': float(np.mean(errors)), 'gesture': gesture, 'variant': variant,
                'blank': blank, 'seconds': horizon / 50,
                'compute_seconds': self.sim.wall_seconds,
                'final_joints': self.sim.body.data.qpos[self.sim.body.qadr].tolist()}


class GestureTrainer:
    def __init__(self, directory, graph, session_factory=GestureSession):
        self.directory, self.graph = Path(directory), Path(graph)
        self.session_factory = session_factory
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.status = {'running': False, 'history': [], 'error': None, 'frame': None,
                       'checkpoint': None, 'phase': 'ready'}

    def publish(self, **values):
        with self.lock:
            self.status.update(values)

    def checkpoints(self):
        return sorted((p.name for p in self.directory.glob('gesture-*.pt')), reverse=True)

    def snapshot(self, include_frame=True):
        with self.lock:
            value = {k: v for k, v in self.status.items() if include_frame or k != 'frame'}
            return copy.deepcopy({**value, 'checkpoints': self.checkpoints(), 'gestures': catalogue()})

    def _load(self, name):
        if not name or Path(name).name != name or not name.startswith('gesture-') or not name.endswith('.pt'):
            raise ValueError('Invalid gesture checkpoint name')
        payload = torch.load(self.directory / name, map_location='cpu', weights_only=True)
        manifest = json.loads((self.graph / 'manifest.json').read_text())
        for key, expected in {'version': VERSION, 'adapter_version': ADAPTER_VERSION,
                              'graph_sha256': manifest['graph_sha256'], 'assets_sha256': asset_digest(),
                              'implementation_sha256': implementation_digest(),
                              'dynamics_version': PAPER_DYNAMICS_VERSION, 'sensory_version': SENSORY_VERSION}.items():
            if payload.get(key) != expected:
                raise ValueError(f'Incompatible gesture checkpoint: {key}')
        params = payload['parameters']
        if not isinstance(params, torch.Tensor) or not torch.isfinite(params).all() or (params.abs() > 4).any():
            raise ValueError('Invalid saved adapter parameters')
        return payload

    def start(self, settings, *, iterations=6, horizon=25, rank=2, seed=42, gesture=None, checkpoint=None):
        if not 1 <= iterations <= 100 or not 5 <= horizon <= 250 or rank not in {1, 2, 4, 8} or not 0 <= seed <= 2147483647:
            raise ValueError('Invalid gesture training settings')
        if gesture is not None and gesture not in GESTURES:
            raise ValueError('Unknown hand gesture')
        # Validate existence and compatibility before launching a worker.
        if not (self.graph / 'manifest.json').exists():
            raise ValueError('Full MaleCNS graph is required')
        saved = self._load(checkpoint) if checkpoint else None
        if saved:
            settings, rank, seed = saved['settings'], saved['rank'], saved['seed']
        with self.lock:
            if self.status['running']:
                raise ValueError('A gesture run is already running')
            self.stop_event.clear()
            self.status = {'running': True, 'mode': 'inference' if gesture else 'training',
                           'phase': 'loading full brain', 'iteration': 0, 'total': iterations,
                           'history': [], 'frame': None, 'error': None, 'checkpoint': checkpoint,
                           'completed_rollouts': 0, 'started_at': time.time(), 'horizon': horizon,
                           'cancelled': False, 'rank': rank,
                           'algorithm': 'Supervised pose loss; SPSA optimizer; constrained low-rank adapter'}
            if gesture and saved:
                for key in ('history', 'before', 'after', 'validation', 'blank_control', 'improved', 'success', 'total'):
                    if key in saved['results']:
                        self.status[key] = copy.deepcopy(saved['results'][key])
            self.thread = threading.Thread(target=self._run,
                args=(copy.deepcopy(settings), iterations, horizon, rank, seed, gesture, saved), daemon=True)
            self.thread.start()

    def _run(self, settings, iterations, horizon, rank, seed, gesture, saved):
        began = time.perf_counter()
        temporary = None
        completed = 0
        try:
            session = self.session_factory(self.graph, settings, rank, seed)
            if self.stop_event.is_set():
                raise Cancelled()
            params = session.adapter.initial.copy()
            if saved:
                if saved['groups'] != [list(g) for g in session.adapter.groups]:
                    raise ValueError('Saved adapter group registry differs')
                params = saved['parameters'].numpy().copy()
            self.publish(adapter=session.adapter.summary())

            def run(p, cue, variant=0, blank=False):
                nonlocal completed
                result = session.rollout(p, cue, horizon, variant, self.stop_event, self.publish, blank=blank)
                completed += 1
                self.publish(completed_rollouts=completed, wall_seconds=time.perf_counter() - began)
                return result

            def evaluate(p, phase, variant=0):
                self.publish(phase=phase)
                results = [run(p, cue, variant) for cue in GESTURES]
                return float(np.mean([r['loss'] for r in results])), results

            if gesture:
                self.publish(phase='frozen adapter inference' if saved else 'untrained baseline inference')
                result = run(params, gesture)
                self.publish(inference=result, phase='inference complete')
                return
            baseline_loss, before = evaluate(params, 'baseline')
            best, best_loss = params.copy(), baseline_loss
            history = [{'iteration': 0, 'loss': baseline_loss, 'best_loss': baseline_loss}]
            self.publish(before=before, history=history.copy())
            rng = np.random.default_rng(seed)
            for step in range(1, iterations + 1):
                self.publish(iteration=step)
                delta = rng.choice([-1., 1.], size=params.shape)
                epsilon = .25 / step ** .101
                plus, _ = evaluate(np.clip(params + epsilon * delta, -4, 4), 'positive perturbation')
                minus, _ = evaluate(np.clip(params - epsilon * delta, -4, 4), 'negative perturbation')
                gradient = np.clip((plus - minus) / (2 * epsilon), -2, 2) * delta
                params = np.clip(params - (.15 / step ** .602) * gradient, -4, 4)
                loss, _ = evaluate(params, 'candidate evaluation')
                if loss < best_loss:
                    best, best_loss = params.copy(), loss
                history.append({'iteration': step, 'loss': loss, 'best_loss': best_loss,
                                'perturbation_difference': plus - minus})
                self.publish(history=history.copy())
            _, after = evaluate(best, 'final training-set evaluation')
            _, held = evaluate(best, 'held-out hand position', variant=1)
            preview = self.snapshot()
            self.publish(phase='blank visual control')
            blank = [run(best, cue, blank=True) for cue in GESTURES]
            response_difference = max(float(np.sqrt(np.mean((np.array(a['final_joints']) - np.array(b['final_joints'])) ** 2)))
                                      for a, b in zip(after, blank))
            trained = best_loss < baseline_loss - 1e-6
            success = trained and all(r['pose_reached'] for r in held) and response_difference > .02
            self.publish(after=after, validation=held, blank_control=blank,
                         visual_response_difference_rad=response_difference,
                         improved=trained, success=success,
                         phase='complete' if success else 'complete — gesture responses not demonstrated')
            self.publish(**{key: preview[key] for key in ('frame', 'current_gesture', 'current_error', 'pose_comparison')})
            session.adapter.apply(best)
            self.publish(adapter=session.adapter.summary())
            payload = {'version': VERSION, 'adapter_version': ADAPTER_VERSION,
                       'dynamics_version': PAPER_DYNAMICS_VERSION, 'sensory_version': SENSORY_VERSION,
                       'graph_sha256': session.sim.full_brain.manifest['graph_sha256'],
                       'assets_sha256': asset_digest(), 'implementation_sha256': implementation_digest(),
                       'settings': settings, 'rank': rank, 'seed': seed,
                       'groups': [list(g) for g in session.adapter.groups],
                       'parameters': torch.from_numpy(best), 'results': self.snapshot(include_frame=False)}
            self.directory.mkdir(parents=True, exist_ok=True)
            name = f'gesture-{time.time_ns()}.pt'
            payload['results'].update(running=False, checkpoint=name)
            temporary = self.directory / (name + '.tmp')
            torch.save(payload, temporary)
            if self.stop_event.is_set():
                raise Cancelled()
            temporary.replace(self.directory / name)
            temporary = None
            self.publish(checkpoint=name)
        except Cancelled:
            self.publish(cancelled=True, phase='stopped — no new checkpoint published')
        except Exception as exc:
            self.publish(error=str(exc), phase='failed')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            self.publish(running=False, wall_seconds=time.perf_counter() - began)


def settings_from_simulation(sim):
    return {'body': asdict(sim.body.parameters),
            'mapping_profile': sim.bridge.mapping_profile if sim.bridge else MAPPING_PROFILE,
            'gains': sim.bridge.parameters.tolist() if sim.bridge else [0.] * 8,
            'senses': asdict(replace(sim.sensors.settings, vision_enabled=True, calibration_sphere=False, vision_model='compound-retina-v1',
                                   hearing_enabled=False, wind_enabled=False))}
