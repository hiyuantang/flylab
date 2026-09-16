"""Small sensor-to-action transformer. No language, connectome, or cue labels.

Sensor schemas and action names are checkpointed. New policies use symmetric
bio-inspired optics (64 groups of 16 per eye); legacy measured schemas remain supported.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch
from torch import nn
from .retina import optical_geometry, BALANCED_MODEL, GRID_MODEL, EMBODIED_MODEL, PATCH_MODEL, SURFACE_MODEL

VERSION = 'sensor-action-transformer-v1'


@dataclass(frozen=True)
class SensorSpec:
    name: str
    samples: int
    channels: int = 1
    tokens: int = 1
    geometry: str = 'vector'


def fly_sensors(tokens=64):
    return [SensorSpec('left_eye', 1024, channels=9, tokens=tokens, geometry='surface-left'),
            SensorSpec('right_eye', 1024, channels=9, tokens=tokens, geometry='surface-right')]


def angular_pool(axes, count):
    """Deterministic farthest-point angular cells; every facet belongs to one cell."""
    centres = [int(np.argmax(axes[:, 0]))]
    distance = np.full(len(axes), np.inf)
    for _ in range(count - 1):
        distance = np.minimum(distance, 1 - axes @ axes[centres[-1]])
        centres.append(int(np.argmax(distance)))
    assignment = (axes @ axes[centres].T).argmax(1)
    matrix = np.zeros((count, len(axes)), dtype=np.float32)
    matrix[assignment, np.arange(len(axes))] = 1
    matrix /= matrix.sum(1, keepdims=True)
    return matrix


class SensorTokenizer(nn.Module):
    def __init__(self, spec, history, width):
        super().__init__()
        self.spec, self.history = spec, history
        if spec.geometry in {'left', 'right', 'balanced-left', 'balanced-right', 'anatomical-left', 'anatomical-right', 'embodied-left', 'embodied-right', 'patch-left', 'patch-right', 'surface-left', 'surface-right'}:
            balanced = spec.geometry.startswith(('balanced-', 'anatomical-', 'embodied-', 'patch-', 'surface-'))
            axes = optical_geometry(SURFACE_MODEL if spec.geometry.startswith('surface-') else PATCH_MODEL if spec.geometry.startswith('patch-') else EMBODIED_MODEL if spec.geometry.startswith('embodied-') else BALANCED_MODEL if spec.geometry.startswith('anatomical-') else GRID_MODEL if balanced else 'compound-retina-v1')[0 if spec.geometry.endswith('left') else 1][0]
            if len(axes) != spec.samples or spec.channels != (9 if spec.geometry.startswith(('patch-', 'surface-')) else 1) or not 1 <= spec.tokens <= spec.samples:
                raise ValueError('Eye schema differs from measured optics')
            self.register_buffer('axes', torch.tensor(axes, dtype=torch.float32))
            if balanced and spec.tokens != 64:
                raise ValueError('Balanced eyes require 64 groups of 16 units')
            pool = np.repeat(np.eye(64, dtype=np.float32), 16, axis=1) / 16 if balanced else angular_pool(axes, spec.tokens)
            self.register_buffer('pool', torch.tensor(pool))
            self.project = nn.Sequential(nn.Linear(history * spec.channels + 3, width), nn.GELU(), nn.Linear(width, width))
        elif spec.geometry == 'vector' and spec.tokens == 1:
            self.project = nn.Linear(history * spec.samples * spec.channels, width)
        else:
            raise ValueError('Unsupported sensor tokenizer')
        self.identity = nn.Parameter(torch.randn(1, spec.tokens, width) * .02)

    def forward(self, value):
        if value.ndim != 4 or tuple(value.shape[1:]) != (self.history, self.spec.samples, self.spec.channels):
            raise ValueError(f'Invalid observation shape for {self.spec.name}')
        if self.spec.geometry == 'vector':
            return self.project(value.flatten(1))[:, None] + self.identity
        samples = value.transpose(1, 2).flatten(2) if self.spec.channels == 9 else value[..., 0].transpose(1, 2)
        samples = torch.cat((samples, self.axes.expand(value.shape[0], -1, -1)), -1)
        return torch.matmul(self.pool, self.project(samples)) + self.identity


class SensorActionTransformer(nn.Module):
    def __init__(self, sensors=None, action_names=None, history=8, chunk=5, width=128, layers=4, heads=4):
        super().__init__()
        self.sensors = sensors or fly_sensors()
        self.action_names = list(action_names or [f'muscle_{i}' for i in range(84)])
        if len({s.name for s in self.sensors}) != len(self.sensors) or not self.action_names:
            raise ValueError('Unique sensor names and nonempty actions required')
        self.history, self.chunk = history, chunk
        self.config = dict(history=history, chunk=chunk, width=width, layers=layers, heads=heads)
        self.tokenizers = nn.ModuleDict({s.name: SensorTokenizer(s, history, width) for s in self.sensors})
        encoder = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0., batch_first=True,
                                              norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        decoder = nn.TransformerDecoderLayer(width, heads, width * 4, dropout=0., batch_first=True,
                                              norm_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder, 2, norm=nn.LayerNorm(width))
        self.queries = nn.Parameter(torch.randn(1, chunk, width) * .02)
        self.output = nn.Linear(width, len(self.action_names))
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.constant_(self.output.bias, -3.)

    def forward(self, observations):
        if set(observations) != set(self.tokenizers):
            raise ValueError('Observation sensor names differ from checkpoint schema')
        encoded = self.encoder(torch.cat([t(observations[name]) for name, t in self.tokenizers.items()], 1))
        decoded = self.decoder(self.queries.expand(encoded.shape[0], -1, -1), encoded)
        return self.output(decoded).sigmoid()

    def schema(self):
        return {'version': VERSION, 'sensors': [asdict(s) for s in self.sensors],
                'actions': self.action_names, 'config': self.config, 'command_seconds': .02}

    @classmethod
    def from_schema(cls, schema):
        if schema.get('version') != VERSION or schema.get('command_seconds') != .02:
            raise ValueError('Unsupported policy schema')
        return cls([SensorSpec(**s) for s in schema['sensors']], schema['actions'], **schema['config'])


def schema_digest(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def eye_observation(frame):
    eyes = frame['vision'].get('eyes', [])
    if len(eyes) != 2 or [e['count'] for e in eyes] not in ([857, 852], [1024, 1024]):
        raise ValueError('Transformer requires the stored compound-eye optical schema')
    return {name: (np.array(eye['patch_channels']['R1-R6'], dtype=np.float32) if frame['vision']['model'] in {PATCH_MODEL, SURFACE_MODEL} else np.array(eye['channels']['R1-R6'], dtype=np.float32)[:, None])
            for name, eye in zip(('left_eye', 'right_eye'), eyes)}


def observation_window(frames, history):
    if not frames:
        raise ValueError('An observation is required')
    selected = ([frames[0]] * max(0, history - len(frames)) + frames)[-history:]
    return {name: np.stack([f[name] for f in selected]) for name in selected[0]}


def policy_vision_model(schema):
    if any(s['geometry'].startswith('surface-') for s in schema['sensors']):
        return SURFACE_MODEL
    if any(s['geometry'].startswith('patch-') for s in schema['sensors']):
        return PATCH_MODEL
    if any(s['geometry'].startswith('embodied-') for s in schema['sensors']):
        return EMBODIED_MODEL
    if any(s['geometry'].startswith('anatomical-') for s in schema['sensors']):
        return BALANCED_MODEL
    return GRID_MODEL if any(s['geometry'].startswith('balanced-') for s in schema['sensors']) else 'compound-retina-v1'
