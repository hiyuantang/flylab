"""A pair of posed 3D hands, shared by Three.js rendering and MuJoCo visual rays.

The hand is a visual stimulus only: its separate ray model cannot exert force
on the fly. Gesture identity is never supplied to a neural input or decoder.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets' / 'meshes' / 'hand'
if not ASSETS.exists():
    ASSETS = Path(__file__).resolve().parents[2] / 'frontend/public/models/hand'
LEGACY_ASSETS = ASSETS
ASSETS = ASSETS / 'facing-v4'
GESTURES = {'palm': 'Stand', 'fist': 'Crouch', 'point': 'Raise left front leg',
            'point_right': 'Raise right front leg', 'point_both': 'Raise both front legs'}
RAISED_FRONT_LEGS = {'point': ('LF',), 'point_right': ('RF',), 'point_both': ('LF', 'RF')}
VERSION = 'hand-facing-scene-v4'
LEGACY_VERSION = 'hand-pair-scene-v3'


def asset_digest():
    digest = hashlib.sha256()
    for name in GESTURES:
        digest.update((ASSETS / f'{name}.obj').read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=10)
def hand_model(gesture, version=VERSION):
    if gesture not in GESTURES:
        raise ValueError('Unknown hand gesture')
    if version not in {VERSION, LEGACY_VERSION}:
        raise ValueError('Unsupported hand layout version')
    assets = ASSETS if version == VERSION else LEGACY_ASSETS
    xml = f'''<mujoco><asset><mesh name="hand" file="{gesture}.obj"/></asset>
    <worldbody><body name="hand" mocap="true"><geom type="mesh" mesh="hand"
    rgba="0.64 0.37 0.24 1" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml, {f'{gesture}.obj': (assets / f'{gesture}.obj').read_bytes()})


class HandStimulus:
    def __init__(self, gesture, *, variant=0, placement=None):
        self.gesture = gesture
        self.version = placement.get('version', VERSION) if placement is not None else VERSION
        self.model = hand_model(gesture, self.version)
        self.data = mujoco.MjData(self.model)
        # Real human-hand scale, at a distance that fits the fly's visual field.
        self.position = np.array([110., 0., 85.]) + np.array([0., variant * 4., 0.])
        self.quaternion = np.array([1., 0., 0., 0.])
        if placement is not None:
            self.position = np.asarray(placement['position'], dtype=float)
            self.quaternion = np.asarray(placement['quaternion'], dtype=float)
            if self.position.shape != (3,) or self.quaternion.shape != (4,) or not np.isfinite(self.position).all() or not np.isfinite(self.quaternion).all() or not np.isclose(np.linalg.norm(self.quaternion), 1):
                raise ValueError('Invalid hand transform')
        self.data.mocap_pos[0] = self.position
        self.data.mocap_quat[0] = self.quaternion
        mujoco.mj_forward(self.model, self.data)

    def sample(self, origin, rays, rgb, depths):
        ids = np.full(len(rays), -1, dtype=np.int32)
        distances = np.full(len(rays), -1., dtype=np.float64)
        mujoco.mj_multiRay(self.model, self.data, origin, np.ascontiguousarray(rays.ravel()),
                          None, True, -1, ids, distances, None, len(rays), 20000.)
        hit = (ids >= 0) & (distances >= 0) & (distances < depths)
        rgb[hit] = self.model.geom_rgba[0, :3]
        depths[hit] = distances[hit]

    def summary(self):
        folder = 'facing-v4/' if self.version == VERSION else ''
        return {'gesture': self.gesture, 'position': self.position.tolist(),
                'quaternion': self.quaternion.tolist(),
                'mesh_url': f'/models/hand/{folder}{self.gesture}.json',
                'focus_offset': [0., 0., 0.] if self.version == VERSION else [-20., 0., 85.],
                'framing_radius': 240. if self.version == VERSION else 200.,
                'color': '#a35e3d', 'version': self.version, 'visual_only': True}


def catalogue():
    return [{'id': key, 'target': label, 'mesh_url': f'/models/hand/facing-v4/{key}.json'}
            for key, label in GESTURES.items()]


def random_placement(rng, position=(0., 0., 0.), heading=0.):
    """Face palms toward the fly; jitter without clipping the floor or crossing it.

    The shared transform is independent of the gesture label. Quaternion is wxyz.
    """
    angles = np.deg2rad(rng.uniform([-4, -4, -5], [4, 4, 5]))
    rotation, yaw, quaternion = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_euler2Quat(rotation, angles, 'xyz')
    mujoco.mju_axisAngle2Quat(yaw, np.array([0., 0., 1.]), heading)
    mujoco.mju_mulQuat(quaternion, yaw, rotation)
    offset = np.array([rng.uniform(100, 125), rng.uniform(-6, 6), rng.uniform(80, 92)])
    # Conservative bounds shared by all cues keep every mesh above the floor and
    # at least 15 mm ahead of the fly, even after randomized rotation.
    corners = hand_bounds()
    rotated_corners = np.empty_like(corners)
    for i, corner in enumerate(corners):
        mujoco.mju_rotVecQuat(rotated_corners[i], corner, rotation)
    offset[0] = max(offset[0], 15. - rotated_corners[:, 0].min())
    offset[2] = max(offset[2], 5. - rotated_corners[:, 2].min())
    rotated = np.empty(3)
    mujoco.mju_rotVecQuat(rotated, offset, yaw)
    return {'position': (np.asarray(position) + rotated).tolist(),
            'quaternion': quaternion.tolist()}


@lru_cache(maxsize=1)
def hand_bounds():
    from itertools import product
    vertices = np.concatenate([np.array(json.loads((ASSETS / f'{cue}.json').read_text())['vertices']).reshape(-1, 3) for cue in GESTURES])
    return np.array(list(product(*zip(vertices.min(axis=0), vertices.max(axis=0)))))
