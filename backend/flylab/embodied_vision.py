"""Engineered facet origins on the fly mesh, with body self-occlusion.

Directions come from the measured-eye interpolation. Registration of those
axes to this different specimen's mesh is approximate, not measured anatomy.
"""
import mujoco
import numpy as np


class SurfaceEye:
    def __init__(self, body, side, axes, patch=False, exposed=False):
        self.model = body.model
        self.geom = body.model.geom(('l', 'r')[side] + '_eye').id
        self.exclude = int(body.model.geom_bodyid[self.geom])
        center = body.data.geom_xpos[self.geom]
        head_rotation = body.data.xmat[body.body_ids['c_head']].reshape(3, 3)
        if exposed:
            from .eye_surface import exposed_origins
            origins = exposed_origins(body, side, axes)
            self.local_offsets = (origins-center) @ body.data.geom_xmat[self.geom].reshape(3, 3)
            return
        placement = axes
        if patch:
            # Register the measured angular map to the outward half of the
            # mesh, independently of viewing axes. Five-degree edge margin.
            # This is an engineered registration, not measured lens positions.
            az = np.arctan2(axes[:, 0], np.abs(axes[:, 1]))
            el = np.arcsin(axes[:, 2])
            az = (az - az.min()) / np.ptp(az) * np.deg2rad(170) - np.deg2rad(85)
            el = (el - el.min()) / np.ptp(el) * np.deg2rad(170) - np.deg2rad(85)
            placement = np.column_stack((np.cos(el) * np.sin(az),
                (1 if side == 0 else -1) * np.cos(el) * np.cos(az), np.sin(el)))
        directions = placement @ head_rotation.T
        distances = np.array([mujoco.mj_rayMesh(body.model, body.data, self.geom, center, ray)
                              for ray in directions])
        if np.any(distances <= 0):
            raise ValueError('Cannot locate every visual unit on the eye surface')
        # Start just outside the source lens. Do not count the observing eye as
        # an opaque occluder; every other visible body segment remains eligible.
        offsets = directions * (distances[:, None] + 1e-5)
        self.local_offsets = offsets @ body.data.geom_xmat[self.geom].reshape(3, 3)

    def cast(self, body, rays):
        if body.model is not self.model:
            raise ValueError('Eye surface cache belongs to a different body model')
        origins = (self.local_offsets @ body.data.geom_xmat[self.geom].reshape(3, 3).T
                   + body.data.geom_xpos[self.geom])
        ids = np.full(len(rays), -1, dtype=np.int32)
        depth = np.full(len(rays), -1., dtype=np.float64)
        # Visible scene geometry and anatomical surface meshes, but no hidden
        # collision proxies. All seven acceptance rays share their facet origin.
        groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        samples = len(rays) // len(origins)
        if samples not in (7, 9) or len(rays) != samples * len(origins):
            raise ValueError("Invalid eye ray count")
        for i, origin in enumerate(origins):
            part = slice(i * samples, (i + 1) * samples)
            mujoco.mj_multiRay(body.model, body.data, origin, np.ascontiguousarray(rays[part].ravel()),
                              groups, True, self.exclude, ids[part], depth[part], None, samples, 20000.)
        return origins, ids, depth
