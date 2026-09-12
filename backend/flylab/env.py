"""Gymnasium access to physical locomotion and direct muscle control.

This task rewards progress toward a target, not odor-label classification.
It exposes engineering control modes; it does not claim measured neural wiring.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from .body import FlyBody, BodyParameters
from .arena import odor_sensors
from .senses import SensorSuite, SensorySettings, sensory_observation


class FlyEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, action_mode="synergy", max_steps=500, task='walk', body_parameters=BodyParameters(), sensory_settings=SensorySettings(), scene_id='lab'):
        if action_mode not in {"synergy", "muscle", "posture"}:
            raise ValueError("action_mode must be synergy, muscle or posture")
        if task not in {'stand', 'walk'}:
            raise ValueError('Unknown physical task')
        self.task = task
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.body = FlyBody(body_parameters, scene_id)
        self.sensors = SensorSuite(sensory_settings)
        self.action_mode = action_mode
        self.max_steps = int(max_steps)
        self.action_space = gym.spaces.Box(-2., 2., (8,), dtype=np.float32) if action_mode == 'posture' else gym.spaces.Box(0., 1., (6 if action_mode == "synergy" else self.body.model.nu,), dtype=np.float32)
        self.target = np.array(self.body.scene['odor_source'])
        self.steps = 0
        self.finished = True
        self.observation_space = gym.spaces.Box(-1., 1., self._observation().shape, dtype=np.float32)

    def _distance(self):
        return float(np.linalg.norm(self.body.data.qpos[:2] - self.target[:2]))

    def _observation(self):
        data = self.body.data
        rotation = data.xmat[self.body.body_ids["c_thorax"]].reshape(3, 3)
        local_target = rotation.T @ np.r_[self.target[:2] - data.qpos[:2], 0.]
        result = np.concatenate([
            np.sin(data.qpos[7:]), np.cos(data.qpos[7:]), np.tanh(data.qvel[6:] / 100),
            data.act * 2 - 1, rotation.ravel(), np.tanh(data.qvel[:6] / 20),
            np.tanh(self.body.foot_feedback() / 10), np.tanh(local_target[:2] / 20),
            odor_sensors(self.body, self.target), np.sin(self.body.phases), np.cos(self.body.phases),
            np.tanh(self.body.magnitudes), [self.steps / self.max_steps],
            sensory_observation(self.sensors.sample(self.body, self.target)),
        ])
        return np.clip(result, -1, 1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.body.reset()
        self.pose_target = self.body.data.qpos[self.body.qadr].copy()
        self.initial_position = self.body.data.qpos[:3].copy()
        options = options or {}
        angle = self.np_random.uniform(-np.pi / 3, np.pi / 3)
        radius = self.np_random.uniform(8, 15)
        target = np.asarray(options.get("target", self.initial_position[:2] + [radius * np.cos(angle), radius * np.sin(angle)]), dtype=float)
        if target.shape not in {(2,), (3,)} or not np.isfinite(target).all() or np.any(np.abs(target) > 10000):
            raise ValueError("target must contain two or three finite coordinates within ±10000 mm")
        self.target = np.r_[target, self.body.scene['spawn'][2] + .01] if target.shape == (2,) else target
        self.steps = 0
        self.finished = False
        return self._observation(), {"distance_mm": self._distance(), "target_mm": self.target[:2].copy(), "action_mode": self.action_mode}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must match action_space with finite values in [0, 1]")
        before = self._distance()
        if self.action_mode == "synergy":
            self.body.step(np.repeat(action, 2))
        elif self.action_mode == 'posture':
            self.body.step_posture(self.pose_target, action)
        else:
            self.body.step_muscles(action)
        self.steps += 1
        distance = self._distance()
        upright = float(self.body.data.xmat[self.body.body_ids["c_thorax"]].reshape(3, 3)[2, 2])
        success = distance < 1. if self.task == 'walk' else False
        fallen = upright < .35 or self.body.data.qpos[2] < self.body.scene['spawn'][2] + .35
        effort = float(np.mean(self.body.data.act ** 2))
        pose_error = float(np.sqrt(np.mean((self.body.data.qpos[self.body.qadr] - self.pose_target) ** 2)))
        height_error = abs(float(self.body.data.qpos[2] - self.initial_position[2]))
        terms = {"progress": before - distance, "effort": -.002 * effort,
                 "success": 5. if success else 0., "fall": -2. if fallen else 0.}
        if self.task == 'stand':
            drift = float(np.linalg.norm(self.body.data.qpos[:2] - self.initial_position[:2]))
            terms = {'upright': .02 * max(0., upright), 'height': -.02 * height_error,
                     'pose': -.02 * pose_error, 'drift': -.01 * drift, 'effort': -.002 * effort, 'fall': -2. if fallen else 0.}
            success = bool(self.steps >= self.max_steps and not fallen and upright > .9 and drift < .5
                           and height_error < .3 and pose_error < .1)
        terminated = bool(success or fallen)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.finished = terminated or truncated
        return self._observation(), float(sum(terms.values())), terminated, truncated, {
            "distance_mm": distance, "upright": upright, "success": success, 'task': self.task,
            'height_error_mm': height_error, 'pose_error_rad': pose_error,
            'reward_version': 'stand-pose-v2' if self.task == 'stand' else 'walk-v1',
            "reward_terms": terms, "simulation_time": float(self.body.data.time),
        }


if "FlyLab-Locomotion-v0" not in gym.registry:
    gym.register("FlyLab-Locomotion-v0", entry_point="flylab.env:FlyEnv")
