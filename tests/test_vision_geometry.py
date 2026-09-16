"""The 3D viewer must use the same ray origins/directions as the sensor."""
import json
from pathlib import Path

import numpy as np
import pytest

from flylab.body import FlyBody
from flylab.embodied_vision import SurfaceEye
from flylab.retina import MODEL, GRID_MODEL, BALANCED_MODEL, EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL, optical_geometry


@pytest.mark.parametrize('model', [MODEL, GRID_MODEL, BALANCED_MODEL, EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL])
def test_displayed_eye_geometry_matches_simulator(model):
    path = Path(__file__).parents[1] / 'frontend/public/models/vision' / (model + '.json')
    value = json.loads(path.read_text())
    assert value['model'] == model
    assert value['frame'] == 'head' and value['units'] == 'mm'
    body = FlyBody()
    head = body.body_ids['c_head']
    rotation = body.data.xmat[head].reshape(3, 3)
    center = body.data.xpos[head]
    for side, (axes, rays) in enumerate(optical_geometry(model)):
        if model in {EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL}:
            origins, _, _ = SurfaceEye(body, side, axes, patch=model == PATCH_MODEL, exposed=model == SURFACE_MODEL).cast(body, rays @ rotation.T)
        else:
            geom = body.model.geom(('l', 'r')[side] + '_eye').id
            origins = np.repeat(body.data.geom_xpos[geom][None], len(axes), axis=0)
        np.testing.assert_allclose(value['eyes'][side]['origins'], (origins - center) @ rotation, atol=1e-12)
        np.testing.assert_array_equal(value['eyes'][side]['directions'], axes)


def test_patch_display_preserves_pixel_order_and_eye_surface():
    import mujoco
    value = json.loads((Path(__file__).parents[1] / 'frontend/public/models/vision' / (PATCH_MODEL + '.json')).read_text())
    body = FlyBody()
    head = body.body_ids['c_head']
    rotation = body.data.xmat[head].reshape(3, 3)
    center = body.data.xpos[head]
    for side, eye in enumerate(value['eyes']):
        points = np.asarray(eye['display_positions']).reshape(1024, 9, 3)
        np.testing.assert_allclose(points[:, 4], eye['origins'], atol=1e-12)
        assert np.isfinite(points).all()
        geom = body.model.geom(('l', 'r')[side] + '_eye').id
        # All nine sample markers lie just outside the actual eye surface.
        for point in points[::64].reshape(-1, 3):
            world = point @ rotation.T + center
            ray = world - body.data.geom_xpos[geom]
            distance = np.linalg.norm(ray)
            ray /= distance
            depth = mujoco.mj_rayMesh(body.model, body.data, geom, body.data.geom_xpos[geom], ray)
            assert abs(distance - depth - 1e-5) < 1e-10
        vertices = np.asarray(eye['surface_vertices'])
        faces = np.asarray(eye['surface_faces'])
        assert faces.min() >= 0 and faces.max() < len(vertices)


def test_exposed_display_has_exact_centers_and_no_buried_shell():
    from flylab.eye_surface import exposed_points
    value = json.loads((Path(__file__).parents[1] / 'frontend/public/models/vision' / (SURFACE_MODEL + '.json')).read_text())
    body = FlyBody()
    head = body.body_ids['c_head']
    rotation = body.data.xmat[head].reshape(3, 3)
    center = body.data.xpos[head]
    for side, eye in enumerate(value['eyes']):
        points = np.asarray(eye['display_positions']).reshape(1024, 9, 3)
        np.testing.assert_allclose(points[:, 4], eye['origins'], atol=1e-12)
        vertices = np.asarray(eye['surface_vertices']) @ rotation.T + center
        faces = np.asarray(eye['surface_faces'])
        assert exposed_points(body, side, vertices[np.unique(faces)], margin=0).all()
