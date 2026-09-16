"""Mesh-exposure registration for engineered retinal origins (not measured lenses)."""
import mujoco
import numpy as np


def exposed_points(body, side, points, margin=.004):
    """Keep surface outside the head, visible toward its lateral exterior.

    A small cross-shaped erosion leaves a head/eye boundary margin. This uses
    the actual head mesh rather than a plane through the eye center.
    """
    head = body.model.geom('c_head').id
    rotation = body.data.xmat[body.body_ids['c_head']].reshape(3, 3)
    ray = rotation @ np.array([0., 1. if side == 0 else -1., 0.])
    offsets = np.array([[0, 0, 0], [margin, 0, 0], [-margin, 0, 0], [0, 0, margin], [0, 0, -margin]]) @ rotation.T
    return np.array([all(mujoco.mj_rayMesh(body.model, body.data, head, point + offset, ray) < 0
                         for offset in offsets) for point in points])


def surface_mesh(body, side):
    geom = body.model.geom(('l', 'r')[side] + '_eye').id
    mesh = int(body.model.geom_dataid[geom])
    start, count = int(body.model.mesh_vertadr[mesh]), int(body.model.mesh_vertnum[mesh])
    vertices = body.model.mesh_vert[start:start+count] @ body.data.geom_xmat[geom].reshape(3, 3).T + body.data.geom_xpos[geom]
    start, count = int(body.model.mesh_faceadr[mesh]), int(body.model.mesh_facenum[mesh])
    return geom, vertices, body.model.mesh_face[start:start+count]


def exposed_origins(body, side, axes):
    geom, vertices, faces = surface_mesh(body, side)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    rng = np.random.default_rng(1407)
    chosen = rng.choice(len(faces), 12000, p=area/area.sum())
    uv = rng.random((len(chosen), 2))
    flip = uv.sum(1) > 1
    uv[flip] = 1-uv[flip]
    t = triangles[chosen]
    points = t[:, 0] + uv[:, :1]*(t[:, 1]-t[:, 0]) + uv[:, 1:]*(t[:, 2]-t[:, 0])
    points = points[exposed_points(body, side, points)]
    if len(points) < len(axes):
        raise ValueError('Insufficient exposed eye surface')
    center = body.data.geom_xpos[geom]
    radial = points-center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    rotation = body.data.xmat[body.body_ids['c_head']].reshape(3, 3)
    # Density follows the measured angular map, with bounded weighting so
    # sparse regions still receive samples. No viewing axes are changed.
    similarity = (radial @ rotation) @ axes.T
    nearest = np.partition(similarity, -6, axis=1)[:, -6:].mean(1)
    density = 1/np.maximum(np.arccos(np.clip(nearest, -1, 1)), .03)
    density = np.clip(np.sqrt(density/np.median(density)), .75, 1.5)
    distance = np.full(len(points), np.inf)
    selected = []
    index = int(np.argmax(density))
    for _ in range(len(axes)):
        selected.append(index)
        distance = np.minimum(distance, np.sum((points-points[index])**2, axis=1))
        distance[selected] = -1
        index = int(np.argmax(distance*density))
    selected_points = points[selected]
    # Match two spatial partitions to retain locality, then restore original
    # axis/group ordering. Spatial position and optical direction remain distinct.
    physical = (selected_points-center) @ rotation
    physical = (physical-physical.mean(0))/np.maximum(np.ptp(physical, axis=0), 1e-8)
    angular = (axes-axes.mean(0))/np.maximum(np.ptp(axes, axis=0), 1e-8)
    result = np.empty_like(selected_points)
    def match(a, b):
        if len(a) == 1:
            result[a[0]] = selected_points[b[0]]
            return
        dim = int(np.argmax(np.ptp(physical[b], axis=0)))
        a = a[np.argsort(angular[a, dim], kind='stable')]
        b = b[np.argsort(physical[b, dim], kind='stable')]
        half = len(a)//2
        match(a[:half], b[:half]); match(a[half:], b[half:])
    match(np.arange(len(axes)), np.arange(len(axes)))
    radial = result-center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    return result + radial*1e-5
