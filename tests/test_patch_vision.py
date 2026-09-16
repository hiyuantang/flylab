"""Patch pixels remain independent; geometry and checkpoints stay explicit."""
from dataclasses import asdict
import numpy as np
import pytest
import torch
from flylab.body import FlyBody, BodyParameters
from flylab.embodied_vision import SurfaceEye
from flylab.retina import PATCH_MODEL, EMBODIED_MODEL, SURFACE_MODEL, optical_geometry
from flylab.senses import SensorSuite, SensorySettings
from flylab.policy_model import SensorActionTransformer, SensorSpec, eye_observation, observation_window, policy_vision_model
from flylab.policy_simulation import PolicySimulation


def test_nine_distinct_rays_preserve_measured_density_and_locality():
    for (axes, rays), (old_axes, _) in zip(optical_geometry(PATCH_MODEL), optical_geometry(EMBODIED_MODEL)):
        np.testing.assert_array_equal(axes, old_axes)
        rays = rays.reshape(1024, 9, 3)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=2), 1)
        np.testing.assert_allclose(rays[:, 4], axes, atol=1e-15)
        dots = axes @ axes.T
        np.fill_diagonal(dots, -1)
        nearest = np.arccos(np.clip(dots.max(1), -1, 1))
        radius = np.arccos(np.clip((rays * axes[:, None]).sum(2), -1, 1))
        assert np.all(radius.max(1) < nearest / 2)
        assert all(len(np.unique(patch, axis=0)) == 9 for patch in rays)


def test_exposed_placement_margin_and_transform():
    body = FlyBody()
    for side, (axes, rays) in enumerate(optical_geometry(PATCH_MODEL)):
        view = SurfaceEye(body, side, axes, patch=True)
        origins, ids, depth = view.cast(body, rays)
        local = (origins - body.data.geom_xpos[view.geom]) @ body.data.xmat[body.body_ids['c_head']].reshape(3, 3)
        assert np.all(local[:, 1] * (1 if side == 0 else -1) > 0)
        # Outward surface extends to dorsal, ventral, front and rear edges.
        az = np.rad2deg(np.arctan2(local[:, 0], abs(local[:, 1])))
        el = np.rad2deg(np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1)))
        np.testing.assert_allclose([az.min(), az.max(), el.min(), el.max()], [-85, 85, -85, 85], atol=1e-8)
        assert len(ids) == 9216 and np.isfinite(depth).all()


def test_patch_values_reach_transformer_without_averaging():
    torch.set_num_threads(1)
    body = FlyBody()
    suite = SensorSuite(SensorySettings(vision_model=PATCH_MODEL, vision_enabled=True))
    frame = suite.sample(body, (0, 0, 1))
    observation = eye_observation(frame)
    assert all(value.shape == (1024, 9) for value in observation.values())
    assert np.any(np.ptp(observation['left_eye'], axis=1) > 0)
    window = {k: torch.from_numpy(v)[None] for k, v in observation_window([observation], 8).items()}
    policy = SensorActionTransformer(width=32, layers=1)
    restored = SensorActionTransformer.from_schema(policy.schema())
    restored.load_state_dict(policy.state_dict())
    torch.testing.assert_close(policy(window), restored(window), rtol=0, atol=0)
    assert policy.tokenizers['left_eye'](window['left_eye']).shape == (1, 64, 32)
    original = window['left_eye'].clone().requires_grad_()
    policy.tokenizers['left_eye'](original).square().sum().backward()
    assert (original.grad.abs().sum((0, 1, 2)) > 0).all()
    assert policy_vision_model(policy.schema()) == SURFACE_MODEL
    with pytest.raises(ValueError, match='observation shape'):
        policy({k: v[..., :1] for k, v in window.items()})


def test_old_schema_still_uses_one_channel_and_old_optics():
    sensors = [SensorSpec('left_eye', 1024, tokens=64, geometry='embodied-left'),
               SensorSpec('right_eye', 1024, tokens=64, geometry='embodied-right')]
    body = FlyBody()
    policy = SensorActionTransformer(sensors=sensors, action_names=[body.model.actuator(i).name for i in range(84)], width=32, layers=1)
    assert policy_vision_model(policy.schema()) == EMBODIED_MODEL
    sim = PolicySimulation(policy, {'body': asdict(BodyParameters()), 'senses': asdict(SensorySettings())})
    sim.advance()
    assert sim.sensors.settings.vision_model == EMBODIED_MODEL
    assert sim.visual_history[0]['left_eye'].shape == (1024, 1)
