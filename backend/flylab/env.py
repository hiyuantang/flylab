"""Gymnasium access to physical locomotion and direct muscle control.

This task rewards progress toward a target, not odor-label classification.
It exposes engineering control modes; it does not claim measured neural wiring.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from .body import FlyBody
from .arena import odor_sensors


class FlyEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, action_mode="synergy", max_steps=500):
        if action_mode not in {"synergy", "muscle"}:
            raise ValueError("action_mode must be synergy or muscle")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.body = FlyBody()
        self.action_mode = action_mode
        self.max_steps = int(max_steps)
        self.action_space = gym.spaces.Box(0., 1., (6 if action_mode == "synergy" else 84,), dtype=np.float32)
        self.target = np.array([12., 0., .01])
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
        ])
        return np.clip(result, -1, 1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.body.reset()
        options = options or {}
        angle = self.np_random.uniform(-np.pi / 3, np.pi / 3)
        radius = self.np_random.uniform(8, 15)
        target = np.asarray(options.get("target", [radius * np.cos(angle), radius * np.sin(angle)]), dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all() or np.any(np.abs(target) > 80):
            raise ValueError("target must contain two finite coordinates within ±80 mm")
        self.target = np.r_[target, .01]
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
        else:
            self.body.step_muscles(action)
        self.steps += 1
        distance = self._distance()
        upright = float(self.body.data.xmat[self.body.body_ids["c_thorax"]].reshape(3, 3)[2, 2])
        success = distance < 1.
        fallen = upright < .35 or self.body.data.qpos[2] < .35
        effort = float(np.mean(self.body.data.act ** 2))
        terms = {"progress": before - distance, "effort": -.002 * effort,
                 "success": 5. if success else 0., "fall": -2. if fallen else 0.}
        terminated = bool(success or fallen)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.finished = terminated or truncated
        return self._observation(), float(sum(terms.values())), terminated, truncated, {
            "distance_mm": distance, "upright": upright, "success": success,
            "reward_terms": terms, "simulation_time": float(self.body.data.time),
        }


if "FlyLab-Locomotion-v0" not in gym.registry:
    gym.register("FlyLab-Locomotion-v0", entry_point="flylab.env:FlyEnv")
