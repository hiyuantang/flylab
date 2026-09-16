import numpy as np
from flylab.body import FlyBody
from flylab.retina import PATCH_MODEL, SURFACE_MODEL, optical_geometry
from flylab.eye_surface import exposed_points, surface_mesh
from flylab.embodied_vision import SurfaceEye
from flylab.policy_model import SensorActionTransformer, policy_vision_model, eye_observation
from flylab.senses import SensorSuite, SensorySettings


def test_exposed_surface_coverage_and_retained_viewing_directions():
    body = FlyBody()
    for side in [0, 1]:
        axes, rays = optical_geometry(SURFACE_MODEL)[side]
        for a, b in zip(optical_geometry(SURFACE_MODEL)[side], optical_geometry(PATCH_MODEL)[side]):
            np.testing.assert_array_equal(a, b)
        eye = SurfaceEye(body, side, axes, exposed=True)
        origins, _, _ = eye.cast(body, rays)
        assert len(np.unique(origins, axis=0)) == 1024
        assert exposed_points(body, side, origins).all()
        geom, vertices, faces = surface_mesh(body, side)
        exposed = vertices[exposed_points(body, side, vertices)]
        distance = np.linalg.norm(exposed[:, None]-origins[None], axis=2).min(1)
        # Independent mesh vertices, not the points used for placement. No
        # large bare regions: every exposed vertex is within 20 micrometers.
        assert distance.max() < .02
        center = body.data.geom_xpos[geom]
        assert np.any((origins[:, 1]-center[1]) * (1 if side == 0 else -1) < 0)


def test_new_surface_schema_and_nine_pixel_observations():
    policy = SensorActionTransformer(width=32, layers=1)
    assert policy_vision_model(policy.schema()) == SURFACE_MODEL
    body = FlyBody()
    frame = SensorSuite(SensorySettings(vision_model=SURFACE_MODEL, vision_enabled=True)).sample(body, (0, 0, 1))
    assert all(x.shape == (1024, 9) for x in eye_observation(frame).values())
