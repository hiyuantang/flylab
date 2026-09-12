"""Explicit synthetic sensory field shared by the workbench and RL environment."""
from __future__ import annotations

import numpy as np


def odor_sensors(body, source, strength=.7):
    """Gaussian concentration sampled at the two moving antennal funiculi (mm)."""
    sites = np.array([body.data.xpos[body.body_ids[f"{side}_funiculus"]] for side in ["l", "r"]])
    distances = np.linalg.norm(sites[:, :2] - np.asarray(source)[:2], axis=1)
    return strength * np.exp(-.5 * (distances / 8.) ** 2)


def bearing_error(body, source):
    rotation = body.data.xmat[body.body_ids["c_thorax"]].reshape(3, 3)
    delta = np.r_[np.asarray(source)[:2] - body.data.qpos[:2], 0.]
    local = rotation.T @ delta
    return float(np.arctan2(local[1], local[0]))
