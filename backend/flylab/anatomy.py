"""Annotation partitions over the original neuron registry, never reduced circuits.

Every child partition includes unknown labels. Index arrays refer to the original
state tensors; cross-partition edges stay in the original count CSR. These source
fields do not claim exclusive neuropil or physiological membership.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json

import numpy as np

LEVELS = ('superclass', 'type', 'somaSide', 'column')
LEVEL_NAMES = {'superclass': 'Superclass', 'type': 'Cell type', 'somaSide': 'Soma side', 'column': 'Assigned hex column', 'neuron': 'Neuron'}


@dataclass(frozen=True)
class NeuronGroup:
    id: str
    label: str
    kind: str
    parent: str | None
    path: tuple
    indices: np.ndarray


def group_id(path):
    if not path:
        return 'all'
    return base64.urlsafe_b64encode(json.dumps(path, separators=(',', ':'), ensure_ascii=True).encode()).decode().rstrip('=')


class AnatomyIndex:
    def __init__(self, brain):
        self.brain = brain
        self.neurons = brain.neurons
        self.ids = brain.ids
        if [r['bodyId'] for r in self.neurons] != list(self.ids):
            raise ValueError('Anatomical annotations do not match state-vector neuron ordering')
        self.groups = {}
        self.children = {}
        self._register((), np.arange(len(self.ids), dtype=np.int64), None)
        self.class_names = sorted({r.get('superclass') for r in self.neurons}, key=lambda x: (x is None, str(x)))
        lookup = {name: i for i, name in enumerate(self.class_names)}
        self.class_index = np.array([lookup[r.get('superclass')] for r in self.neurons])
        self.ptr = np.load(brain.directory / 'indptr.npy', mmap_mode='r')
        self.pre = np.load(brain.directory / 'indices.npy', mmap_mode='r')
        self.counts = np.load(brain.directory / 'counts.npy', mmap_mode='r')
        self._wiring_cache = OrderedDict()
        with (brain.directory / 'neurons.feather').open('rb') as source:
            self.annotation_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()

    def _register(self, path, indices, parent):
        identifier = group_id(path)
        if identifier in self.groups:
            return self.groups[identifier]
        kind, value = path[-1] if path else ('all', 'All imported neurons')
        label = str(value) if value is not None else 'Unassigned'
        if kind == 'somaSide':
            label = {'L': 'Left soma', 'R': 'Right soma', 'M': 'Midline soma', None: 'Unassigned soma side'}.get(value, str(value))
        indices.setflags(write=False)
        group = NeuronGroup(identifier, label, kind, parent, path, indices)
        self.groups[identifier] = group
        return group

    def resolve(self, identifier):
        if identifier in self.groups:
            return self.groups[identifier]
        # Resolve bookmarks independently of which parents have been visited.
        try:
            if len(identifier) > 2048:
                raise ValueError('Oversized group ID')
            path = json.loads(base64.urlsafe_b64decode(identifier + '=' * (-len(identifier) % 4)))
            if not isinstance(path, list) or len(path) > len(LEVELS) + 1:
                raise ValueError('Invalid group path')
            current = self.groups['all']
            for part in path:
                if not isinstance(part, list) or len(part) != 2 or not isinstance(part[0], str) or (part[1] is not None and not isinstance(part[1], (str, int))):
                    raise ValueError('Invalid group component')
                kind, value = part
                if kind == 'neuron':
                    if part != path[-1] or not isinstance(value, int):
                        raise ValueError('Invalid neuron path')
                    pos = self.brain.lookup.get(value)
                    if pos is None or pos not in current.indices:
                        raise ValueError('Neuron is outside the selected group')
                    current = self._register(current.path + (('neuron', value),), np.array([pos]), current.id)
                else:
                    candidates = self.partition(current.id)
                    current = next(g for g in candidates if g.path[-1] == (kind, value))
            if current.id != identifier:
                raise ValueError('Noncanonical group ID')
            return current
        except (ValueError, TypeError, StopIteration, UnicodeError) as exc:
            raise ValueError('Unknown anatomical group') from exc

    def partition(self, identifier='all'):
        """Disjoint child index views whose union is exactly the parent indices."""
        group = self.resolve(identifier)
        if group.id not in self.children:
            depth = len(group.path)
            buckets = {}
            if depth < len(LEVELS) and group.kind != 'neuron':
                kind = LEVELS[depth]
                for i in group.indices:
                    row = self.neurons[i]
                    value = row.get(kind)
                    if kind == 'column':
                        a, b = row.get('assignedOlHex1'), row.get('assignedOlHex2')
                        value = f'{a:g}, {b:g}' if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None
                    buckets.setdefault(value, []).append(i)
                self.children[group.id] = [self._register(group.path + ((kind, value),), np.array(indices, dtype=np.int64), group.id)
                                           for value, indices in sorted(buckets.items(), key=lambda x: (x[0] is None, str(x[0]).casefold()))]
            else:
                self.children[group.id] = []
        return self.children[group.id]

    def describe(self, group, rates):
        activity = rates[group.indices]
        return {'id': group.id, 'label': group.label, 'kind': group.kind, 'kind_label': LEVEL_NAMES.get(group.kind, 'Nervous system'),
                'count': len(group.indices), 'mean_hz': float(activity.mean()) if len(activity) else 0.,
                'active_neurons': int((activity > 1.).sum()), 'parent': group.parent}

    def wiring(self, identifier):
        """Exact internal and crossing counts, grouped by the other superclass."""
        group = self.resolve(identifier)
        if identifier in self._wiring_cache:
            self._wiring_cache.move_to_end(identifier)
            return self._wiring_cache[identifier]
        member = np.zeros(len(self.ids), dtype=bool)
        member[group.indices] = True
        m = len(self.class_names)
        # Columns: incoming pairs/synapses, outgoing pairs/synapses.
        crossing = np.zeros((m, 4), dtype=np.int64)
        internal = np.zeros(2, dtype=np.int64)
        for first in range(0, len(self.ids), 4096):
            last = min(first + 4096, len(self.ids))
            begin, end = int(self.ptr[first]), int(self.ptr[last])
            pre, counts = self.pre[begin:end], self.counts[begin:end]
            post = np.repeat(np.arange(first, last), np.diff(self.ptr[first:last + 1]))
            incoming, outgoing = member[post] & ~member[pre], member[pre] & ~member[post]
            inside = member[pre] & member[post]
            internal += [int(inside.sum()), int(counts[inside].sum())]
            for mask, others, col in [(incoming, pre, 0), (outgoing, post, 2)]:
                categories = self.class_index[others[mask]]
                crossing[:, col] += np.bincount(categories, minlength=m)
                np.add.at(crossing[:, col + 1], categories, counts[mask])
        result = {'internal': {'connections': int(internal[0]), 'synapses': int(internal[1])},
                  'incoming': {'connections': int(crossing[:, 0].sum()), 'synapses': int(crossing[:, 1].sum())},
                  'outgoing': {'connections': int(crossing[:, 2].sum()), 'synapses': int(crossing[:, 3].sum())},
                  'routes': [{'label': name or 'Unassigned', 'incoming_connections': int(row[0]), 'incoming_synapses': int(row[1]),
                              'outgoing_connections': int(row[2]), 'outgoing_synapses': int(row[3])}
                             for name, row in zip(self.class_names, crossing) if row.any()]}
        self._wiring_cache[identifier] = result
        if len(self._wiring_cache) > 32:
            self._wiring_cache.popitem(last=False)
        return result

    def snapshot(self, identifier='all', view='groups', query='', offset=0, limit=30):
        group = self.resolve(identifier)
        rates = self.brain.rates.detach().cpu().numpy()
        trail = []
        current = group
        while current:
            trail.append(self.describe(current, rates))
            current = self.groups.get(current.parent)
        children = self.partition(group.id)
        query = query.casefold().strip()
        items = []
        if view == 'groups':
            matches = [g for g in children if query in g.label.casefold()]
            total = len(matches)
            items = [self.describe(g, rates) for g in matches[offset:offset + limit]]
        elif view == 'neurons':
            matches = [int(i) for i in group.indices if not query or query in str(self.ids[i]) or query in str(self.neurons[i].get('type') or '').casefold()]
            total = len(matches)
            for i in matches[offset:offset + limit]:
                row = self.neurons[i]
                path = group.path if group.kind == 'neuron' else group.path + (('neuron', int(self.ids[i])),)
                items.append({'id': group_id(path), 'body_id': int(self.ids[i]), 'type': row.get('type'), 'superclass': row.get('superclass'),
                              'soma_side': row.get('somaSide'), 'transmitter': row.get('transmitter'),
                              'hex': [row.get('assignedOlHex1'), row.get('assignedOlHex2')],
                              'rate_hz': float(rates[i]), 'voltage': float(self.brain.voltage[i]),
                              'voltage_unit': 'mV' if self.brain.config.profile == 'shiu-2024' else 'normalized'})
        else:
            total = 0
        return {'schema_version': 1, 'group': self.describe(group, rates), 'breadcrumbs': trail[::-1],
                'next_kind': LEVEL_NAMES.get(LEVELS[len(group.path)]) if len(group.path) < len(LEVELS) and group.kind != 'neuron' else None,
                'view': view, 'items': items, 'total': total, 'offset': offset, 'limit': limit,
                'wiring': self.wiring(identifier) if view == 'wiring' else None,
                'neural_time_ms': self.brain.time_ms, 'graph_sha256': self.brain.manifest['graph_sha256'],
                'annotation_sha256': self.annotation_sha256,
                'source': 'MaleCNS imported neurons.feather: superclass, type, somaSide, assignedOlHex1/2',
                'membership': 'Disjoint annotation partitions over original neuron indices; no neurons merged or edges removed.',
                'limitations': 'Superclass is not a neuropil boundary. Soma side is not arbor territory. Assigned hex coordinates are incomplete; neuropil overlap is not yet mapped.'}
