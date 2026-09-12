"""Reproducible black-box reinforcement learning on actual MuJoCo rollouts.

Cross-entropy search trains eight bounded controller log gains. Connectome mode
changes neural efficacy and peripheral gains; posture mode calibrates an explicit
feedback baseline. Neither changes anatomical synapse counts.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import tempfile
import numpy as np
import torch
from .env import FlyEnv
from .body import BodyParameters, FlyBody
from .full_connectome import FullBrain, Physiology, DYNAMICS_VERSION
from .neuromuscular import NeuromuscularBridge
from .motor_mapping import MAPPING_PROFILE, PROFILES, validate_body_profile
from .physical_policy import TrialEnvironment, validate_policy
from .senses import SensorSuite, SensorySettings, SENSORY_VERSION


def rollout(env, parameters, seed, bridge=None, stop=None, perturb=False, environment=TrialEnvironment()):
    if env.body.scene_id != environment.scene_id:
        env.body = FlyBody(env.body.parameters, environment.scene_id)
    env.sensors = SensorSuite(environment.senses)
    env.reset(seed=seed, options={'target': environment.source})
    if bridge:
        bridge.brain.reset()
        bridge.set_parameters(parameters)
    reward = 0.
    minimum_upright = 1.
    max_penetration = 0.
    activation_sum = 0.
    final = {}
    for step in range(env.max_steps):
        if stop and stop.is_set():
            return None
        if perturb and step == env.max_steps // 2:
            env.body.data.xfrc_applied[env.body.body_ids['c_thorax'], 1] = 5.
        frame = env.sensors.sample(env.body, env.target, environment.intensity if environment.odor != 'none' else 0., environment.mode == 'spatial')
        action = bridge.command(env.body, env.target, frame=frame) if bridge else parameters
        _, r, done, truncated, final = env.step(np.asarray(action, dtype=np.float32))
        env.body.data.xfrc_applied[:] = 0
        reward += r
        activation_sum += float(env.body.data.act.mean())
        minimum_upright = min(minimum_upright, final['upright'])
        for contact in env.body.data.contact:
            if env.body.model.geom_bodyid[contact.geom1] and env.body.model.geom_bodyid[contact.geom2]:
                max_penetration = max(max_penetration, -float(contact.dist))
        if done or truncated:
            break
    return {'reward': reward, 'steps': env.steps, 'seconds': float(env.body.data.time),
            'success': final.get('success', False), 'minimum_upright': minimum_upright,
            'displacement_mm': float(np.linalg.norm(env.body.data.qpos[:2] - env.initial_position[:2])),
            'distance_mm': final.get('distance_mm'), 'max_sampled_penetration_mm': max_penetration,
            'mean_muscle_activation': activation_sum / max(1, env.steps),
            'neural_spikes': bridge.brain.total_spikes if bridge else None,
            'height_error_mm': final.get('height_error_mm'), 'pose_error_rad': final.get('pose_error_rad'),
            'reward_version': final.get('reward_version'),
            'perturbation': '5 µN lateral force for 20 ms at midpoint' if perturb else 'none'}


class PhysicalTrainer:
    def __init__(self, directory, graph_directory):
        self.directory = Path(directory)
        self.graph_directory = Path(graph_directory)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.status = {'running': False, 'history': [], 'error': None, 'checkpoint': None}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def start(self, mode='posture', task='stand', generations=4, population=6, horizon=50, seed=42,
              body_parameters=BodyParameters(), physiology=Physiology.paper(), sensory_settings=SensorySettings(vision_enabled=True), environment=None, mapping_profile=MAPPING_PROFILE):
        if mode == 'connectome':
            validate_body_profile(mapping_profile, body_parameters)
        if mapping_profile not in PROFILES:
            raise ValueError("Unknown muscle mapping profile")
        environment = environment or TrialEnvironment(senses=sensory_settings)
        if mode not in {'posture', 'connectome'} or task not in {'stand', 'walk'}:
            raise ValueError('Unknown physical controller/task')
        if not (1 <= generations <= 100 and 4 <= population <= 32 and 5 <= horizon <= 500 and 0 <= seed < 2**31):
            raise ValueError('Invalid training budget or seed')
        if mode == 'connectome' and not (self.graph_directory / 'manifest.json').is_file():
            raise ValueError('Build the full graph first')
        with self.lock:
            if self.status['running']:
                raise ValueError('Physical training is already running')
            self.stop_event.clear()
            self.status = {'running': True, 'mode': mode, 'task': task, 'generation': 0, 'total': generations,
                           'completed_rollouts': 0, 'horizon': horizon, 'history': [], 'before': None, 'after': None,
                           'validation': None, 'checkpoint': None, 'error': None, 'cancelled': False,
                           'mapping_profile': mapping_profile, 'environment': environment.to_dict(), 'seed': seed, 'algorithm': 'Cross-entropy search on eight bounded log gains'}
            self.thread = threading.Thread(target=self._run, args=(mode, task, generations, population, horizon, seed, body_parameters, physiology, environment, mapping_profile), daemon=True)
            self.thread.start()

    def _run(self, mode, task, generations, population, horizon, seed, body_parameters, physiology, environment, mapping_profile):
        try:
            env = FlyEnv(action_mode='muscle' if mode == 'connectome' else 'posture', task=task,
                         max_steps=horizon, body_parameters=body_parameters, scene_id=environment.scene_id)
            bridge = NeuromuscularBridge(FullBrain(self.graph_directory, physiology), mapping_profile) if mode == 'connectome' else None
            rng = np.random.default_rng(seed)
            mean, std = np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32) * .4
            before = rollout(env, mean, seed, bridge, self.stop_event, environment=environment)
            if before is None:
                return
            best, best_score = mean.copy(), before['reward']
            with self.lock:
                self.status['before'] = before
            for generation in range(generations):
                candidates = np.clip(rng.normal(mean, std, (population, 8)), -2, 2).astype(np.float32)
                candidates[0] = best
                scores = []
                for candidate in candidates:
                    result = rollout(env, candidate, seed, bridge, self.stop_event, environment=environment)
                    if result is None:
                        return
                    scores.append(result['reward'])
                    if result['reward'] > best_score:
                        best, best_score = candidate.copy(), result['reward']
                    with self.lock:
                        self.status['completed_rollouts'] += 1
                elite = candidates[np.argsort(scores)[-max(2, population // 3):]]
                mean = elite.mean(axis=0)
                std = np.maximum(elite.std(axis=0), .08)
                with self.lock:
                    self.status['generation'] = generation + 1
                    self.status['history'].append({'generation': generation + 1, 'reward': best_score, 'mean_reward': float(np.mean(scores))})
            after = rollout(env, best, seed, bridge, self.stop_event, environment=environment)
            if after is None:
                return
            passive = FlyEnv(action_mode='muscle', task=task, max_steps=horizon, body_parameters=body_parameters, scene_id=environment.scene_id)
            passive.reset(seed=seed, options={'target': environment.source})
            passive_reward = 0.
            for _ in range(horizon):
                if self.stop_event.is_set():
                    return
                _, reward, done, truncated, info = passive.step(np.zeros(passive.body.model.nu, dtype=np.float32))
                passive_reward += reward
                if done or truncated:
                    break
            ablation = {'reward': passive_reward, 'success': info['success'], 'seconds': float(passive.body.data.time),
                        'description': 'All muscle excitations zero; passive springs and contacts remain.'}
            # A distinct perturbation and altered mass probe generalization; these
            # are held-out conditions, not reused selection scores.
            validation = []
            for mass, perturb in [(1., True), (.9, False), (1.1, False)]:
                params = BodyParameters(**{**asdict(body_parameters), 'mass_scale': body_parameters.mass_scale * mass})
                test_env = FlyEnv(action_mode=env.action_mode, task=task, max_steps=horizon, body_parameters=params, scene_id=environment.scene_id)
                result = rollout(test_env, best, seed + 100, bridge, self.stop_event, perturb, environment)
                if result is None:
                    return
                validation.append({'mass_multiplier': mass, **result})
            self.directory.mkdir(parents=True, exist_ok=True)
            name = f'{mode}-{task}-{time.time_ns()}.pt'
            payload = {'format': 'flylab-physical-v1', 'mode': mode, 'task': task, 'parameters': torch.from_numpy(best),
                       'seed': seed, 'body_parameters': asdict(body_parameters), 'physiology': asdict(physiology),
                       'mapping_profile': mapping_profile,
                       'graph_sha256': bridge.brain.manifest['graph_sha256'] if bridge else None,
                       'before': before, 'after': after, 'validation': validation, 'passive_ablation': ablation,
                       'training': {'generations': generations, 'population': population, 'horizon': horizon},
                       'reward_version': after['reward_version'],
                       'environment': environment.to_dict(), 'sensory_version': SENSORY_VERSION,
                       'dynamics_version': bridge.brain.dynamics_version if bridge else None,
                       'assumptions': 'Optimization against engineering task rewards; not fitting biological recordings. Anatomical counts fixed.'}
            # Readers only see the final name after serialization succeeds.
            with tempfile.TemporaryDirectory(prefix='.checkpoint-', dir=self.directory) as tmp:
                pending = Path(tmp) / 'policy.tmp'
                torch.save(payload, pending)
                with self.lock:
                    if self.stop_event.is_set():
                        return
                    pending.replace(self.directory / name)
                    self.status.update(after=after, validation=validation, passive_ablation=ablation, checkpoint=name, parameter_change=float(np.linalg.norm(best)))
        except Exception as exc:
            with self.lock:
                self.status['error'] = str(exc)
        finally:
            with self.lock:
                self.status['running'] = False
                self.status['cancelled'] = self.stop_event.is_set() and self.status['checkpoint'] is None

    def load(self, name):
        if Path(name).name != name or not name.endswith('.pt'):
            raise ValueError('Invalid checkpoint name')
        result = torch.load(self.directory / name, weights_only=True, map_location='cpu')
        validate_policy(result)
        return result

    def checkpoints(self):
        return sorted([p.name for p in self.directory.glob('*.pt')], reverse=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['posture', 'connectome'], default='posture')
    parser.add_argument('--task', choices=['stand', 'walk'], default='stand')
    parser.add_argument('--generations', type=int, default=4)
    parser.add_argument('--population', type=int, default=6)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    trainer = PhysicalTrainer(args.data / 'physical-checkpoints', args.data / 'full')
    trainer.start(mode=args.mode, task=args.task, generations=args.generations,
                  population=args.population, horizon=args.horizon, seed=args.seed)
    try:
        trainer.thread.join()
    except KeyboardInterrupt:
        trainer.stop_event.set()
        trainer.thread.join()
    print(json.dumps(trainer.snapshot(), indent=2))
    if trainer.snapshot()['error']:
        raise SystemExit(1)
