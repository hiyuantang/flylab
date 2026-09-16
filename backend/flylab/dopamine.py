"""Experimental, annotation-gated mushroom-body teaching; no graph pruning.

See docs/DOPAMINE.md for the biological evidence and assumed rate/learning law.
Anatomical CSR values remain immutable; only separate effective weights change.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch

PROFILE = 'mb-dopamine-v1'
PATHWAYS = (('positive', 'PAM01', 'MBON01', 'gamma5'),
            ('negative', 'PPL101', 'MBON11', 'gamma1pedc'))
PULSE_SECONDS = .2
INPUT_HZ = 120.
TRACE_SECONDS = .5
RELEASE_SECONDS = .2
RATE_SCALE_HZ = 100.
LEARNING_PER_SECOND = 1.
MIN_FACTOR = .2


class DopamineCircuit:
    def __init__(self, brain):
        self.brain = brain
        ptr = brain.weights.crow_indices().numpy()
        pres = brain.weights.col_indices().numpy()
        # Raw integer counts, independent of the fast-transmitter sign policy.
        counts = np.load(brain.directory / 'counts.npy', mmap_mode='r')
        rows = brain.neurons
        self.kc = np.array([i for i, r in enumerate(rows)
                            if r.get('class') == 'Kenyon_Cell'
                            and str(r.get('type')).startswith('KCg')
                            and r.get('transmitter') == 'acetylcholine'], dtype=np.int64)
        kc_lookup = {int(k): i for i, k in enumerate(self.kc)}
        self.dans = [np.array([i for i, r in enumerate(rows)
                              if r.get('type') == dan and r.get('transmitter') == 'dopamine'], dtype=np.int64)
                     for _, dan, _, _ in PATHWAYS]
        self.dan = np.concatenate(self.dans)
        dan_lookup = {int(k): i for i, k in enumerate(self.dan)}
        kc_routes = np.zeros((len(self.kc), len(self.dan)))
        for k, post in enumerate(self.kc):
            for e in range(ptr[post], ptr[post + 1]):
                d = dan_lookup.get(int(pres[e]))
                if d is not None:
                    kc_routes[k, d] = counts[e]
        edges, kc_indices, channels, routes, outgoing = [], [], [], [], []
        out_ptr, out_post = brain.out_ptr.numpy(), brain.out_post.numpy()
        for channel, (_, _, mbon, _) in enumerate(PATHWAYS):
            group = np.array([dan_lookup[int(d)] for d in self.dans[channel]], dtype=np.int64)
            for post, row in enumerate(rows):
                if row.get('type') != mbon:
                    continue
                incoming = set(int(i) for i in pres[ptr[post]:ptr[post + 1]])
                # Require the same DAN to contact this KC and this MBON. This
                # establishes a measured motif, not a reconstructed receptor site.
                allowed = [d for d in group if int(self.dan[d]) in incoming]
                for e in range(ptr[post], ptr[post + 1]):
                    pre = int(pres[e])
                    k = kc_lookup.get(pre)
                    if k is None:
                        continue
                    route = np.zeros(len(self.dan))
                    route[allowed] = kc_routes[k, allowed]
                    if not route.sum():
                        continue
                    edges.append(e)
                    kc_indices.append(k)
                    channels.append(channel)
                    routes.append(route / route.sum())
                    start, end = out_ptr[pre:pre + 2]
                    offset = int(start + np.searchsorted(out_post[start:end], post))
                    if offset >= end or out_post[offset] != post:
                        raise ValueError('Dopamine edge index disagrees with outgoing graph')
                    outgoing.append(offset)
        self.edges = torch.tensor(edges, dtype=torch.int64)
        self.outgoing = torch.tensor(outgoing, dtype=torch.int64)
        self.edge_kc = np.array(kc_indices, dtype=np.int64)
        self.channels = np.array(channels, dtype=np.int64)
        self.routes = np.array(routes, dtype=np.float64).reshape(len(edges), len(self.dan)) if len(self.dan) else np.zeros((0, 0))
        self.base = brain.weights.values()[self.edges].clone()
        digest = hashlib.sha256(PROFILE.encode())
        for value in (brain.ids[self.dan], brain.ids[self.kc], self.edges.numpy(), self.routes):
            digest.update(value.tobytes())
        self.mapping_sha256 = digest.hexdigest()
        self.available = all(len(d) and np.any(self.channels == c) for c, d in enumerate(self.dans))
        self.factors = np.ones(len(edges))
        self.eligibility = np.zeros(len(self.kc))
        self.release = np.zeros(len(edges))
        self.dan_rates = np.zeros(2)
        self.baseline = np.zeros(len(self.dan))
        self.enabled = False
        self.score = 0.
        self.remaining = 0.
        self.events = 0
        self.last_event = None
        self._metal = None
        self._gpu_edges = None
        self._input = None

    def submit(self, score, time):
        if not self.available:
            raise ValueError('Both annotated dopamine pathways and their measured KC/MBON motifs are required')
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError('Reward score must be finite and between -1 and 1')
        if score == 0:
            return
        if self.remaining > 1e-9:
            raise ValueError('A teaching pulse is still active; advance simulation or stop teaching first')
        self.baseline = self.brain.rates[torch.as_tensor(self.dan, device=self.brain.device)].detach().cpu().numpy().astype(np.float64)
        self.enabled = True
        self.score, self.remaining = float(score), PULSE_SECONDS
        self.events += 1
        self.last_event = {'score': float(score), 'time': float(time)}
        self._input = torch.zeros(len(self.brain.ids), dtype=torch.float64)
        self._input[self.dans[0 if score > 0 else 1]] = abs(score) * INPUT_HZ

    def stop(self):
        self.enabled = False
        self.remaining = 0.
        self._input = None
        self.release.fill(0)

    def input_rates(self, duration):
        if not self.enabled or self.remaining <= 1e-9:
            return None
        # A clock change can leave a partial final command interval. Preserve
        # expected event count by averaging its remaining exposure over that cycle.
        return self._input * min(1., self.remaining / duration)

    def update(self, rates, duration):
        if not self.available:
            return
        rates = rates.numpy() if isinstance(rates, torch.Tensor) else rates
        kc = np.clip(rates[self.kc].astype(np.float64) / RATE_SCALE_HZ, 0, 1)
        self.eligibility += (kc - self.eligibility) * -math.expm1(-duration / TRACE_SECONDS)
        self.dan_rates = np.array([rates[d].astype(np.float64).mean() for d in self.dans])
        if not self.enabled:
            return
        gain = self.brain.output_gain[torch.as_tensor(self.dan, device=self.brain.device)].detach().cpu().numpy()
        da = np.zeros(len(self.dan))
        if self.remaining > 1e-9:
            # This minimal external-teaching model only attributes the selected
            # channel's phasic increase to feedback. Tonic/endogenous modulation
            # is out of scope; otherwise high baseline firing teaches both valences.
            selected = np.isin(self.dan, self.dans[0 if self.score > 0 else 1])
            da[selected] = np.clip((rates[self.dan[selected]].astype(np.float64) - self.baseline[selected])
                                   * gain[selected] / RATE_SCALE_HZ, 0, 1)
            da *= min(1., self.remaining / duration)
        local = self.routes @ da
        self.release += (local - self.release) * -math.expm1(-duration / RELEASE_SECONDS)
        self.factors *= np.exp(-LEARNING_PER_SECOND * duration * self.eligibility[self.edge_kc] * self.release)
        np.maximum(self.factors, MIN_FACTOR, out=self.factors)
        self.apply_weights()
        self.remaining = max(0., self.remaining - duration)
        if self.remaining <= 1e-9:
            self.remaining, self._input = 0., None

    def apply_weights(self):
        if not len(self.edges):
            return
        values = self.base * torch.from_numpy(self.factors)
        self.brain.out_weight[self.outgoing] = values
        metal = self.brain.metal
        if metal is not None:
            if metal is not self._metal:
                self._metal = metal
                self._gpu_edges = self.edges.to(device=metal.weight.device)
            metal.weight[self._gpu_edges] = values.to(device=metal.weight.device, dtype=metal.weight.dtype)

    def reset(self):
        self.stop()
        self.factors.fill(1)
        self.eligibility.fill(0)
        self.dan_rates.fill(0)
        self.baseline.fill(0)
        self.score, self.events, self.last_event = 0., 0, None
        self.apply_weights()

    def summary(self):
        return {'profile': PROFILE, 'available': self.available, 'enabled': self.enabled,
                'pulse_remaining_s': self.remaining, 'last_event': self.last_event, 'events': self.events,
                'plastic_edges': len(self.edges), 'changed_edges': int(np.sum(self.factors < 1 - 1e-6)),
                'pathways': [
                    {'valence': valence, 'dan_type': dan, 'mbon_type': mbon, 'compartment': compartment,
                     'neurons': len(self.dans[c]), 'dan_hz': float(self.dan_rates[c]),
                     'plastic_edges': int(np.sum(self.channels == c)),
                     'mean_factor': float(self.factors[self.channels == c].mean()) if np.any(self.channels == c) else 1.}
                    for c, (valence, dan, mbon, compartment) in enumerate(PATHWAYS)]}

    def state_dict(self):
        state = {'profile': PROFILE, 'mapping_sha256': self.mapping_sha256, 'enabled': self.enabled,
                'score': self.score, 'remaining': self.remaining, 'events': self.events, 'last_event': self.last_event,
                **{k: torch.from_numpy(getattr(self, k).copy())
                   for k in ('factors', 'eligibility', 'release', 'dan_rates', 'baseline')}}
        self._validate_state(state, self.brain.time_ms / 1000.)
        return state

    def _validate_state(self, state, time):
        if not self.available or state['profile'] != PROFILE or state['mapping_sha256'] != self.mapping_sha256:
            raise ValueError('Dopamine profile or anatomical mapping differs')
        if (type(state['enabled']) is not bool or not math.isfinite(state['score']) or not -1 <= state['score'] <= 1
                or not math.isfinite(state['remaining']) or not 0 <= state['remaining'] <= PULSE_SECONDS
                or type(state['events']) is not int or state['events'] < 0
                or (state['remaining'] > 0 and (not state['enabled'] or state['score'] == 0))):
            raise ValueError('Invalid dopamine teaching state')
        event = state['last_event']
        if (event is None) != (state['events'] == 0):
            raise ValueError('Invalid dopamine event history')
        if event is not None and (not math.isfinite(event['time']) or not 0 <= event['time'] <= time
                                  or event['score'] != state['score'] or event['score'] == 0):
            raise ValueError('Invalid dopamine event timestamp or score')
        values = {}
        for key in ('factors', 'eligibility', 'release', 'dan_rates', 'baseline'):
            value = state[key]
            low, high = (MIN_FACTOR, 1.) if key == 'factors' else (0., math.inf if key in {'dan_rates', 'baseline'} else 1.)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                    or value.shape != getattr(self, key).shape or not torch.isfinite(value).all()
                    or torch.any((value < low) | (value > high))):
                raise ValueError(f'Invalid dopamine state: {key}')
            values[key] = value.numpy().copy()
        return values

    def load_state(self, state, time):
        values = self._validate_state(state, time)
        for key, value in values.items():
            setattr(self, key, value)
        for key in ('enabled', 'score', 'remaining', 'events', 'last_event'):
            setattr(self, key, state[key])
        self._input = torch.zeros(len(self.brain.ids), dtype=torch.float64) if self.remaining > 0 else None
        if self._input is not None:
            self._input[self.dans[0 if self.score > 0 else 1]] = abs(self.score) * INPUT_HZ
        self.apply_weights()
