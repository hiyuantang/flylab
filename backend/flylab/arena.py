"""Explicit synthetic sensory field shared by the workbench and RL environment."""
from __future__ import annotations

import numpy as np


def odor_sensors(body, source, strength=.7, *, spatial_model='legacy-v2'):
    """Gaussian concentration sampled at the two moving antennal funiculi (mm)."""
    sites = np.array([body.data.xpos[body.body_ids[f"{side}_funiculus"]] for side in ["l", "r"]])
    if spatial_model not in {'legacy-v2', 'geometry-v3'}:
        raise ValueError('Unknown spatial sensor model')
    dimensions = 3 if spatial_model == 'geometry-v3' else 2
    distances = np.linalg.norm(sites[:, :dimensions] - np.asarray(source)[:dimensions], axis=1)
    return strength * np.exp(-.5 * (distances / 8.) ** 2)


def bearing_error(body, source):
    rotation = body.data.xmat[body.body_ids["c_thorax"]].reshape(3, 3)
    delta = np.r_[np.asarray(source)[:2] - body.data.qpos[:2], 0.]
    local = rotation.T @ delta
    return float(np.arctan2(local[1], local[0]))
