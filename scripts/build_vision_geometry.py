"""Generate visualization origins from the same optical code used by sensors.
Run: PYTHONPATH=backend .venv/bin/python scripts/build_vision_geometry.py
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import mujoco
from flylab.body import FlyBody

def project_to_surface(points, triangles):
    """Closest triangle projection; radial projection can hit the wrong fold."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    norm2 = np.sum(normal*normal, axis=1)
    d00, d01, d11 = np.sum(ab*ab, axis=1), np.sum(ab*ac, axis=1), np.sum(ac*ac, axis=1)
    denominator = np.maximum(d00*d11-d01*d01, 1e-30)
    result = []
    for start in range(0, len(points), 64):
        p = points[start:start+64, None]
        v = p-a
        plane = p - (np.sum(v*normal, axis=2)/np.maximum(norm2, 1e-30))[..., None]*normal
        v = plane-a
        d20, d21 = np.sum(v*ab, axis=2), np.sum(v*ac, axis=2)
        u, w = (d11*d20-d01*d21)/denominator, (d00*d21-d01*d20)/denominator
        distance = np.sum((plane-p)**2, axis=2)
        distance[(u < 0) | (w < 0) | (u+w > 1) | (norm2 < 1e-25)] = np.inf
        best = plane.copy()
        for x, y in ((a,b), (b,c), (c,a)):
            edge = y-x
            fraction = np.clip(np.sum((p-x)*edge, axis=2)/np.maximum(np.sum(edge*edge, axis=1), 1e-30), 0, 1)
            projected = x + fraction[..., None]*edge
            candidate = np.sum((projected-p)**2, axis=2)
            use = candidate < distance
            best[use], distance[use] = projected[use], candidate[use]
        index = np.argmin(distance, axis=1)
        result.extend(best[np.arange(len(index)), index])
    return np.asarray(result)


@lru_cache(maxsize=6)
def vision_geometry(model):
    """Read-only optical placement in head coordinates, matching sensor origins."""
    from flylab.retina import optical_geometry, EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL
    from flylab.embodied_vision import SurfaceEye
    body = FlyBody()
    head = body.body_ids['c_head']
    rotation = body.data.xmat[head].reshape(3, 3)
    center = body.data.xpos[head]
    eyes = []
    for side, (axes, rays) in enumerate(optical_geometry(model)):
        if model in {EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL}:
            view = SurfaceEye(body, side, axes, patch=model == PATCH_MODEL, exposed=model == SURFACE_MODEL)
            origins = (view.local_offsets @ body.data.geom_xmat[view.geom].reshape(3, 3).T
                       + body.data.geom_xpos[view.geom])
        else:
            geom = body.model.geom(('l', 'r')[side] + '_eye').id
            origins = np.repeat(body.data.geom_xpos[geom][None], len(axes), axis=0)
        geom = body.model.geom(('l', 'r')[side] + '_eye').id
        mesh = int(body.model.geom_dataid[geom])
        begin, count = int(body.model.mesh_vertadr[mesh]), int(body.model.mesh_vertnum[mesh])
        vertices = body.model.mesh_vert[begin:begin+count] @ body.data.geom_xmat[geom].reshape(3, 3).T + body.data.geom_xpos[geom]
        begin, count = int(body.model.mesh_faceadr[mesh]), int(body.model.mesh_facenum[mesh])
        faces = body.model.mesh_face[begin:begin+count]
        surface = origins
        if model not in {EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL}:
            # Display-only surface projection for historical center-origin eyes.
            view = SurfaceEye(body, side, axes)
            surface = view.local_offsets @ body.data.geom_xmat[geom].reshape(3, 3).T + body.data.geom_xpos[geom]
        display = surface
        if model in {PATCH_MODEL, SURFACE_MODEL}:
            # Separate pixel positions for visualization only. Keep all nine
            # training rays at their true shared facet origin.
            radial = surface - body.data.geom_xpos[geom]
            radial /= np.linalg.norm(radial, axis=1, keepdims=True)
            tangent = np.cross(radial, [0., 0., 1.])
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            other = np.cross(radial, tangent)
            distance = np.linalg.norm(surface[:, None] - surface[None], axis=2)
            np.fill_diagonal(distance, np.inf)
            spacing = distance.min(1) / 3
            display = []
            for i in range(len(surface)):
                for y in (-1, 0, 1):
                    for x in (-1, 0, 1):
                        if model == SURFACE_MODEL:
                            display.append(surface[i] + spacing[i] * (x*tangent[i] + y*other[i]))
                            continue
                        ray = surface[i] - body.data.geom_xpos[geom] + spacing[i] * (x*tangent[i] + y*other[i])
                        ray /= np.linalg.norm(ray)
                        depth = mujoco.mj_rayMesh(body.model, body.data, geom, body.data.geom_xpos[geom], ray)
                        if depth <= 0:
                            raise ValueError('Display pixel missed eye mesh')
                        display.append(body.data.geom_xpos[geom] + ray * (depth + 1e-5))
            display = np.asarray(display)
            if model == SURFACE_MODEL:
                display = project_to_surface(display, vertices[faces])
                outward = display-body.data.geom_xpos[geom]
                outward /= np.linalg.norm(outward, axis=1, keepdims=True)
                display += outward*1e-5
                display.reshape(-1, 9, 3)[:, 4] = surface
        if model == SURFACE_MODEL:
            from flylab.eye_surface import exposed_points
            visible = exposed_points(body, side, vertices, margin=0)
            # The buried attachment is not a retina. Display the exposed cap.
            faces = faces[visible[faces].all(1)]
        eyes.append({'surface_vertices': ((vertices-center) @ rotation).tolist(),
                     'surface_faces': faces.tolist(),
                     'display_positions': ((display-center) @ rotation).tolist(),
                     'origins': ((origins - center) @ rotation).tolist(), 'directions': axes.tolist()})
    return {'model': model, 'frame': 'head', 'units': 'mm', 'eyes': eyes}


if __name__ == '__main__':
    folder = Path('frontend/public/models/vision')
    folder.mkdir(parents=True, exist_ok=True)
    for model in ['compound-retina-v1', 'compound-retina-balanced-v1', 'compound-retina-balanced-v2', 'compound-retina-balanced-v3', 'compound-retina-balanced-v4', 'compound-retina-balanced-v5']:
        (folder / (model + '.json')).write_text(json.dumps(vision_geometry(model), separators=(',', ':')) + '\n')
