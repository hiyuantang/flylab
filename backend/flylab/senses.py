"""Deterministic, headless sensory sampling of the shared virtual arena.

Rays intersect the same static MuJoCo primitives rendered in the web scene.
The laboratory contains no virtual calibration object by default. New embodied transformer optics include fly self-occlusion; legacy optics omit it.
Transparent optics and photometric shading are omitted. Angular sampling, hearing tuning and transduction gains are hypotheses.
"""
from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np

SENSORY_VERSION = 'scene-senses-v2'
EYE_WIDTH, EYE_HEIGHT = 16, 8
BEACON_RADIUS = .7


@dataclass(frozen=True)
class SensorySettings:
    vision_model: str = "legacy-grid-v2"
    proprioception_model: str = 'legacy-position-v1'
    contact_model: str = 'legacy-leg-v1'
    spatial_model: str = 'legacy-v2'
    calibration_sphere: bool = False
    vision_enabled: bool = False
    hearing_enabled: bool = False
    wind_enabled: bool = False
    touch_enabled: bool = True
    proprioception_enabled: bool = True
    illumination: float = 1.
    sound_amplitude: float = .8
    sound_frequency: float = 250.
    wind_speed: float = 20.
    wind_direction: float = 0.
    stimulus_position: tuple[float, float, float] = (4., 2., 1.5)

    def __post_init__(self):
        if self.spatial_model not in {'legacy-v2', 'geometry-v3'}:
            raise ValueError('Unknown spatial sensor model')
        if self.contact_model not in {'legacy-leg-v1', 'tarsal-contact-v1'}:
            raise ValueError('Unknown contact model')
        if self.proprioception_model not in {'legacy-position-v1', 'feco-opponent-v1'}:
            raise ValueError('Unknown proprioception model')
        if self.vision_model not in {'legacy-grid-v2', 'compound-retina-v1', 'compound-retina-balanced-v1', 'compound-retina-balanced-v2', 'compound-retina-balanced-v3', 'compound-retina-balanced-v4', 'compound-retina-balanced-v5'}:
            raise ValueError('Unknown vision model')
        for name in ('calibration_sphere', 'vision_enabled', 'hearing_enabled', 'wind_enabled', 'touch_enabled', 'proprioception_enabled'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name, low, high in [('illumination', 0, 1), ('sound_amplitude', 0, 1), ('sound_frequency', 50, 1000),
                                ('wind_speed', 0, 100), ('wind_direction', -180, 180)]:
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid {name}')
        p = self.stimulus_position
        if len(p) != 3 or not all(math.isfinite(v) for v in p) or max(abs(p[0]), abs(p[1])) > 10000 or not .7 <= p[2] <= 5000:
            raise ValueError('Stimulus must have x/y within ±10000 mm and height 0.7–5000 mm')


class SensorSuite:
    def __init__(self, settings=SensorySettings()):
        self.settings = settings
        self.visual_object = None
        self.surface_eyes = {}
        if settings.vision_model in {"compound-retina-v1", "compound-retina-balanced-v1", "compound-retina-balanced-v2", "compound-retina-balanced-v3", "compound-retina-balanced-v4", "compound-retina-balanced-v5"}:
            from .retina import optical_geometry
            self.directions = [rays for _, rays in optical_geometry(settings.vision_model)]
            return
        azimuth, elevation = np.meshgrid(np.linspace(75, -75, EYE_WIDTH), np.linspace(50, -50, EYE_HEIGHT))
        self.directions = []
        for yaw in (60, -60):
            a, e = np.deg2rad(azimuth + yaw), np.deg2rad(elevation)
            self.directions.append(np.stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)), axis=-1).reshape(-1, 3))

    def sample(self, body, odor_source, intensity=.7, spatial=False, duration_s=.02):
        from .arena import odor_sensors
        s, data = self.settings, body.data
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Sensory interval must be positive and finite')
        corrected = s.spatial_model == 'geometry-v3'
        rotation = data.xmat[body.body_ids['c_head']].reshape(3, 3)
        source = np.asarray(s.stimulus_position)
        retinas, distances, hearing, wind = [], [], [], []
        eyes = []
        compound = s.vision_model in {"compound-retina-v1", "compound-retina-balanced-v1", "compound-retina-balanced-v2", "compound-retina-balanced-v3", "compound-retina-balanced-v4", "compound-retina-balanced-v5"}
        # RMS over the configured controller interval; analytic integration
        # avoids sampling the acoustic carrier at one instant per command.
        omega, dt, time = 2 * math.pi * s.sound_frequency, duration_s, float(data.time)
        carrier = math.sqrt(max(0., 1 - (math.sin(2 * omega * (time + dt)) - math.sin(2 * omega * time)) / (2 * omega * dt)))
        tuning = math.exp(-.5 * (math.log2(s.sound_frequency / 250.) / .75) ** 2)
        angle = math.radians(s.wind_direction)
        air = np.array([math.cos(angle), math.sin(angle), 0.]) * s.wind_speed - data.qvel[:3]
        gravity = rotation.T @ np.array([0., 0., -1.])
        if corrected:
            g = body.model.opt.gravity
            gravity = rotation.T @ (g / np.linalg.norm(g)) if np.linalg.norm(g) else np.zeros(3)
        jacobian = np.zeros((3, body.model.nv)) if corrected and s.wind_enabled else None
        for side, name in enumerate(('l', 'r')):
            # Mesh origins coincide; MuJoCo geom centers locate the two eye surfaces.
            eye = data.geom_xpos[mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, f'{name}_eye')]
            rays = self.directions[side] @ rotation.T
            patch = s.vision_model in {'compound-retina-balanced-v4', 'compound-retina-balanced-v5'}
            samples = 9 if patch else 7
            embodied = s.vision_model in {'compound-retina-balanced-v3', 'compound-retina-balanced-v4', 'compound-retina-balanced-v5'}
            if embodied:
                from .embodied_vision import SurfaceEye
                from .retina import optical_geometry
                if side not in self.surface_eyes or self.surface_eyes[side].model is not body.model:
                    self.surface_eyes[side] = SurfaceEye(body, side, optical_geometry(s.vision_model)[side][0], patch=patch, exposed=s.vision_model == 'compound-retina-balanced-v5')
                origins, ids, depth = self.surface_eyes[side].cast(body, rays)
                ray_origins = np.repeat(origins, samples, axis=0)
            else:
                ids = np.full(len(rays), -1, dtype=np.int32)
                depth = np.full(len(rays), -1., dtype=np.float64)
                # Preserve the original world-only optics for existing weights.
                mujoco.mj_multiRay(body.model, data, eye, np.ascontiguousarray(rays.ravel()),
                                  np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1,
                                  ids, depth, None, len(rays), 20000.)
            valid = ids >= 0
            distances_world = np.where(valid, depth, np.inf)
            background = body.scene['background']
            background_rgb = np.array([int(background[i:i+2], 16) / 255 for i in (1, 3, 5)])
            rgb = np.tile(background_rgb, (len(rays),1))
            rgb[valid] = body.model.geom_rgba[ids[valid], :3]
            if body.scene_id == 'lab':
                ground = ids == 0
                rgb[ground] = .58
                points = (ray_origins[ground] if embodied else eye) + rays[ground] * depth[ground, None]
                grid = np.any(np.abs((points[:, :2] + .25) % .5 - .25) < .015, axis=1)
                rgb[np.flatnonzero(ground)[grid]] = .4
                if s.calibration_sphere:
                    offset = (ray_origins if embodied else eye) - source
                    b = np.sum(rays * offset, axis=1) if embodied else rays @ offset
                    squared = np.sum(offset * offset, axis=1) if embodied else offset @ offset
                    discriminant = b * b - (squared - BEACON_RADIUS ** 2)
                    sphere = np.full(len(rays), np.inf)
                    hit = discriminant >= 0
                    near = -b[hit] - np.sqrt(discriminant[hit])
                    far = -b[hit] + np.sqrt(discriminant[hit])
                    sphere[hit] = np.where(near > 0, near, np.where(far > 0, far, np.inf))
                    rgb[sphere < distances_world] = .04
                    distances_world = np.minimum(sphere, distances_world)
            if self.visual_object is not None:
                if embodied:
                    for i, origin in enumerate(origins):
                        part = slice(i * samples, (i + 1) * samples)
                        self.visual_object.sample(origin, rays[part], rgb[part], distances_world[part])
                else:
                    self.visual_object.sample(eye, rays, rgb, distances_world)
            if compound:
                from .retina import integrate_facets, receptor_channels, optical_geometry
                # Convert sRGB material values to linear visible-band intensity.
                rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                facets = (rgb.reshape(-1, 9, 3).mean(1) if patch else integrate_facets(rgb)) * s.illumination * s.vision_enabled
                channels = receptor_channels(facets)
                axes = optical_geometry(s.vision_model)[side][0]
                angles = np.rad2deg(np.column_stack((np.arctan2(axes[:,1], axes[:,0]), np.arcsin(axes[:,2]))))
                eyes.append({'sampling': 'bio-inspired' if s.vision_model.startswith('compound-retina-balanced-') else 'measured', 'count': len(axes), 'angles_degrees': angles.tolist(),
                             'channels': {k:v.tolist() for k,v in channels.items()}})
                if patch:
                    pixels = receptor_channels(rgb * s.illumination * s.vision_enabled)
                    eyes[-1]['patch_channels'] = {k: v.reshape(-1, 9).tolist() for k, v in pixels.items()}
                    patch_axes = optical_geometry(s.vision_model)[side][1]
                    eyes[-1]['patch_angles_degrees'] = np.rad2deg(np.column_stack((np.arctan2(patch_axes[:,1], patch_axes[:,0]), np.arcsin(patch_axes[:,2])))).tolist()
                retinas.append(channels['R1-R6'][None,:])
            else:
                pixels = rgb @ [.2126, .7152, .0722]
                retinas.append((pixels * s.illumination * s.vision_enabled).reshape(EYE_HEIGHT, EYE_WIDTH))
            nearest = distances_world.min()
            distances.append(float(nearest) if np.isfinite(nearest) else None)
            antenna = data.xpos[body.body_ids[f'{name}_funiculus']]
            local_air = air
            if jacobian is not None:
                # Velocity at the receptor, including root rotation and every
                # upstream joint. qvel[:3] alone omits omega cross radius.
                mujoco.mj_jac(body.model, data, jacobian, None, antenna,
                              body.body_ids[f'{name}_funiculus'])
                local_air = np.array([math.cos(angle), math.sin(angle), 0.]) * s.wind_speed - jacobian @ data.qvel
            delta = source - antenna
            distance = np.linalg.norm(delta)
            axis = rotation @ np.array([.5, (1 if side == 0 else -1) * math.sqrt(.75), 0.])
            directionality = .2 + .8 * abs(float(delta @ axis)) / max(distance, 1e-8)
            hearing.append(float(np.clip(s.sound_amplitude * tuning * carrier * directionality / (1 + (distance / 8) ** 2), 0, 1)) if s.hearing_enabled else 0.)
            # The scalar JO tuning remains a hypothesis. The new tilt proxy
            # distinguishes inversion from upright and vanishes in zero gravity.
            tilt = (np.linalg.norm(gravity - [0., 0., -1.]) if np.any(gravity) else 0.) if corrected else np.linalg.norm(gravity[:2])
            wind.append(float(np.clip(abs(local_air @ axis) / 100 + .25 * tilt, 0, 1)) if s.wind_enabled else 0.)
        odor = odor_sensors(body, odor_source, intensity, spatial_model=s.spatial_model) if spatial else np.full(2, intensity)
        contact_force = body.foot_feedback() if s.contact_model == 'tarsal-contact-v1' else body.legacy_leg_feedback()
        contact = np.clip(contact_force / 5, 0, 1) * s.touch_enabled
        angles = data.qpos[body.qadr].reshape(6, 7)
        position = (.5 + .5 * np.sin(angles[:, 5])) * s.proprioception_enabled
        images = [r.tolist() for r in retinas]
        from .leg_feedback import sample_legs
        legs = sample_legs(body, s.proprioception_enabled) if s.proprioception_model == 'feco-opponent-v1' else None
        return {'version': SENSORY_VERSION, 'time': float(data.time), 'settings': asdict(s), 'scene_id': body.scene_id,
                'legs': legs,
                'vision': {'model': s.vision_model, 'width': EYE_WIDTH, 'height': EYE_HEIGHT, 'pixels': images,
                           'eyes': eyes, 'mean': [float(r.mean()) for r in retinas], 'nearest_surface_mm': distances,
                           'optics': '1024 units per eye with nine separate pixels. Origins cover the eye mesh exposed outside the head, with a 0.004 mm boundary margin. Measured nonuniform viewing directions retained. Surface registration and RGB receptors are engineered approximations.' if s.vision_model == 'compound-retina-balanced-v5' else '1024 units per eye, each with nine separate point samples; 64 tokens per eye. Measured nonuniform viewing directions; engineered registration over outward eye surface with a five-degree margin. Body occlusion enabled. RGB proxies, not biological photoreceptors; no UV or active retinal motion.' if s.vision_model == 'compound-retina-balanced-v4' else 'Measured-map directions; estimated eye-surface origins; body and leg occlusion enabled. 1024 units per eye, 64 groups of 16. Symmetry, mesh alignment and 2 degree acceptance ring are assumptions. RGB proxy; UV, polarization and active retinal motion are not modeled.' if s.vision_model == 'compound-retina-balanced-v3' else 'Measured eye-map interpolation; 1024 units per eye, 64 groups of 16. Mirrored symmetry and 2 degree acceptance ring are assumptions; not measured anatomy.' if s.vision_model == 'compound-retina-balanced-v2' else '1024 units per eye; 64 groups of 16. Assumed symmetry and 2 degree acceptance ring; not measured anatomy.' if s.vision_model.startswith('compound-retina-balanced-') else 'Measured microCT directions; assumed head alignment and 2 degree acceptance ring. RGB visible-band proxy; UV and polarization unavailable.' if compound else 'Legacy angular grid'},
                'hearing': hearing, 'wind': wind, 'gravity': gravity.tolist(),
                'odor': odor.tolist(), 'touch': contact.tolist(), 'proprioception': position.tolist()}


def sensory_observation(frame):
    """Version-dependent visible samples plus four antennal channels."""
    return np.r_[np.concatenate([np.asarray(p).ravel() for p in frame['vision']['pixels']]), frame['hearing'], frame['wind']].astype(np.float32)
