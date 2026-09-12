"""Validated physical policy settings shared by training and the workbench."""
from dataclasses import asdict, dataclass, field
import math

import torch

from .body import BodyParameters
from .full_connectome import Physiology
from .senses import SensorySettings
from .scenes import get_scene
from .motor_mapping import PROFILES, LEGACY_PROFILE, validate_body_profile


@dataclass(frozen=True)
class TrialEnvironment:
    mode: str = 'uniform'
    odor: str = 'A'
    intensity: float = .7
    source: tuple[float, float, float] = (12., 0., .01)
    senses: SensorySettings = field(default_factory=SensorySettings)
    scene_id: str = 'lab'
    scene_version: str | None = None

    def __post_init__(self):
        version = get_scene(self.scene_id)['version']
        if self.scene_version is not None and self.scene_version != version:
            raise ValueError('Checkpoint scene geometry version does not match the installed scene')
        object.__setattr__(self, 'scene_version', version)
        if isinstance(self.senses, dict):
            object.__setattr__(self, 'senses', SensorySettings(**self.senses))
        if not isinstance(self.senses, SensorySettings):
            raise ValueError('Invalid sensory settings')
        if self.mode not in {'uniform', 'spatial'} or self.odor not in {'A', 'B', 'none'}:
            raise ValueError('Invalid trial environment')
        if not math.isfinite(self.intensity) or not 0 <= self.intensity <= 1:
            raise ValueError('Invalid trial odor intensity')
        if len(self.source) != 3 or not all(math.isfinite(x) and abs(x) <= 10000 for x in self.source) or not 0 <= self.source[2] <= 5000:
            raise ValueError('Invalid trial source coordinates')

    def to_dict(self):
        return {**asdict(self), 'source': list(self.source)}


def validate_policy(payload):
    """Validate the complete application contract before changing live state."""
    try:
        if payload['format'] != 'flylab-physical-v1' or payload['mode'] not in {'posture', 'connectome'} or payload['task'] not in {'stand', 'walk'}:
            raise ValueError('Invalid physical checkpoint format, controller or task')
        values = payload['parameters']
        if not isinstance(values, torch.Tensor) or values.shape != (8,) or not values.is_floating_point() or not torch.isfinite(values).all() or torch.any(values.abs() > 2):
            raise ValueError('Invalid policy parameters')
        body = BodyParameters(**payload['body_parameters'])
        physiology = Physiology(**payload['physiology'])
        if payload['mode'] == 'connectome' and not isinstance(payload['graph_sha256'], str):
            raise ValueError('Missing anatomical graph fingerprint')
        if payload.get('mapping_profile', LEGACY_PROFILE) not in PROFILES:
            raise ValueError('Unknown muscle mapping profile')
        if payload['mode'] == 'connectome':
            validate_body_profile(payload.get('mapping_profile', LEGACY_PROFILE), body)
        if payload['mode'] == 'connectome' and not math.isclose(round(20 / physiology.dt_ms) * physiology.dt_ms, 20):
            raise ValueError('Neural integration time must divide the 20 ms body step')
        # Earlier checkpoints used these same fixed training settings implicitly.
        environment = TrialEnvironment(**payload.get('environment', {}))
        return values.detach().cpu().float().numpy().copy(), body, physiology, environment
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Incomplete or malformed physical checkpoint') from exc
