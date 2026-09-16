"""Bounded, type-shared low-rank modulation of existing synaptic efficacies.

W' = W * (1 + .75 tanh(U[group(post)] @ V[group(pre)] / sqrt(rank))).
This LoRA-inspired multiplicative adaptation is not standard additive LoRA.
All topology, signs and source counts remain unchanged. Canonical base values
stay on CPU; applying an adapter also updates the active Metal weight buffer.
"""
import math
import numpy as np
import torch

VERSION = 'connectome-low-rank-v1'


class ConnectomeAdapter:
    def __init__(self, brain, rank=2, seed=42, grouping="superclass-side-v1"):
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 64:
            raise ValueError('Rank must be an integer from 1 to 64')
        if brain.config.profile != 'shiu-2024':
            raise ValueError('Adapter requires paper-profile physiology')
        if grouping not in {'superclass-side-v1', 'motor-target-v2'}:
            raise ValueError('Unknown adapter grouping')
        self.brain, self.rank, self.grouping = brain, rank, grouping
        labels = [(str(n.get('superclass') or 'unknown'), str(n.get('rootSide') or 'unknown'))
                  for n in brain.neurons]
        if grouping == 'motor-target-v2':
            # Refine the existing partition, so legacy factors can be inherited
            # exactly. VNC rootSide is usually absent; the motor bridge uses somaSide.
            labels = [(*label, '|'.join(str(n.get(key) or 'unknown')
                       for key in ('somaSide', 'subclass', 'type'))
                       if n.get('superclass') in {'vnc_motor', 'cb_motor'} else '')
                      for label, n in zip(labels, brain.neurons)]
        self.groups = sorted(set(labels))
        lookup = {key: i for i, key in enumerate(self.groups)}
        group = np.array([lookup[key] for key in labels], dtype=np.int64)
        self.group_count = len(self.groups)
        rng = np.random.default_rng(seed)
        self.initial = np.stack([np.zeros((len(self.groups), rank)),
                                 rng.normal(0, .2, (len(self.groups), rank))])
        self.parameters = self.initial.copy()
        self.base = brain.weights.values().clone()
        self.base_out = brain.out_weight.clone()
        ptr = brain.weights.crow_indices().numpy()
        posts = np.repeat(group, np.diff(ptr))
        pres = group[brain.weights.col_indices().numpy()]
        self.pairs = (posts * len(self.groups) + pres).astype(np.int32)
        out_pre = np.repeat(group, np.diff(brain.out_ptr.numpy()))
        self.out_pairs = (group[brain.out_post.numpy()] * len(self.groups) + out_pre).astype(np.int32)

    def inherit(self, groups, parameters):
        """Expand a legacy partition without changing any effective edge weight."""
        lookup = {tuple(group): i for i, group in enumerate(groups)}
        current = [tuple(group) for group in self.groups]
        if all(group in lookup for group in current):
            indices = [lookup[group] for group in current]
        elif self.grouping == 'motor-target-v2' and all(len(group) == 2 for group in lookup):
            indices = [lookup[group[:2]] for group in current]
        else:
            raise ValueError('Saved adapter group registry differs')
        return parameters[:, indices, :]

    @torch.no_grad()
    def apply(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != self.initial.shape or not np.isfinite(parameters).all() or np.max(np.abs(parameters)) > 4:
            raise ValueError('Invalid bounded adapter parameters')
        u, v = parameters
        factors = (1 + .75 * np.tanh(u @ v.T / math.sqrt(self.rank))).ravel()
        for target, base, pairs in ((self.brain.weights.values(), self.base, self.pairs),
                                    (self.brain.out_weight, self.base_out, self.out_pairs)):
            for start in range(0, len(pairs), 262144):
                end = start + 262144
                target[start:end].copy_(base[start:end] * torch.from_numpy(factors[pairs[start:end]]))
        if self.brain.metal is not None:
            target = self.brain.metal.weight
            target.copy_(self.brain.weights.values().to(device=target.device, dtype=target.dtype))
        self.parameters = parameters.copy()

    def summary(self):
        u, v = self.parameters
        factors = 1 + .75 * np.tanh(u @ v.T / math.sqrt(self.rank))
        return {'version': VERSION, 'grouping': self.grouping, 'rank': self.rank, 'groups': self.group_count,
                'trainable_parameters': self.parameters.size,
                'parameter_change': float(np.linalg.norm(self.parameters - self.initial)),
                'min_factor': float(factors.min()), 'max_factor': float(factors.max()),
                'neurons': len(self.brain.ids), 'edges': len(self.base)}
