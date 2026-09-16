from dataclasses import replace

import mujoco
import numpy as np

from flylab.body import FlyBody
from flylab.embodied_vision import SurfaceEye
from flylab.retina import EMBODIED_MODEL, BALANCED_MODEL, optical_geometry
from flylab.senses import SensorSuite, SensorySettings


def test_rear_blind_region_and_lateral_rear_coverage():
    # Check near the horizon, where azimuth is meaningful. Do not incorrectly
    # clip dorsal facets to the same azimuth bounds as equatorial facets.
    axes = optical_geometry(EMBODIED_MODEL)[0][0]
    elevation = np.rad2deg(np.arcsin(axes[:, 2]))
    azimuth = np.rad2deg(np.arctan2(axes[:, 1], axes[:, 0]))
    horizon = azimuth[np.abs(elevation) < 10]
    assert 145 < horizon.max() < 165
    assert -15 < horizon.min() < 0
    assert not np.any(np.abs(horizon) > 165)


def test_surface_origins_body_occlusion_and_head_attachment():
    body = FlyBody()
    axes, rays = optical_geometry(EMBODIED_MODEL)[0]
    view = SurfaceEye(body, 0, axes)
    origins, ids, depths = view.cast(body, rays)
    eye = body.data.geom_xpos[view.geom]
    assert origins.shape == (1024, 3)
    assert np.all(np.linalg.norm(origins - eye, axis=1) > .08)
    names = {body.model.geom(int(g)).name for g in ids if g >= 0}
    assert 'lf_tibia' in names
    assert 'l_arista' in names
    assert 'l_eye' not in names
    assert (depths[ids >= 0] >= 0).all()
    # Anatomical legs actually block the rays: hide only these test meshes and
    # verify those exact rays now reach a farther surface or the background.
    leg_ids = [body.model.geom(n).id for n in names if n.startswith(('lf_', 'lm_', 'lh_'))]
    leg_hits = np.isin(ids, leg_ids)
    body.model.geom_group[leg_ids] = 2
    _, clear_ids, clear_depth = view.cast(body, rays)
    assert leg_hits.any()
    assert np.all((clear_ids[leg_hits] < 0) | (clear_depth[leg_hits] >= depths[leg_hits] - 1e-8))
    assert np.any(clear_ids[leg_hits] != ids[leg_hits])
    body.model.geom_group[leg_ids] = 1
    pivot = body.data.qpos[:3].copy()
    body.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    mujoco.mj_forward(body.model, body.data)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated, _, _ = view.cast(body, rays @ rotation.T)
    np.testing.assert_allclose(rotated, (origins - pivot) @ rotation.T + pivot, atol=1e-8)


def test_distant_stimulus_cannot_overwrite_body_and_old_optics_remain_distinct():
    class DistantStimulus:
        def __init__(self):
            self.visible = []
        def sample(self, origin, rays, rgb, depths):
            visible = depths > 100
            self.visible.extend(visible.tolist())
            rgb[visible] = .9
            depths[visible] = 100
    body = FlyBody()
    settings = SensorySettings(vision_model=EMBODIED_MODEL, vision_enabled=True)
    suite = SensorSuite(settings)
    plain = suite.sample(body, [0, 0, 0])
    stimulus = DistantStimulus()
    suite.visual_object = stimulus
    with_stimulus = suite.sample(body, [0, 0, 0])
    assert len(stimulus.visible) == 2 * 1024 * 7
    assert not all(stimulus.visible)
    assert with_stimulus['vision']['eyes'] != plain['vision']['eyes']
    legacy = SensorSuite(replace(settings, vision_model=BALANCED_MODEL))
    assert legacy.sample(body, [0, 0, 0])['vision']['eyes'] != plain['vision']['eyes']
    assert not legacy.surface_eyes
