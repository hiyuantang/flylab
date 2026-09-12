"""All annotated MaleCNS neurons, exact count CSR, and stateful sparse dynamics.

Full means the induced graph of every annotation with a non-null superclass.
Unclassified segments are excluded explicitly, not declared non-neuronal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from .paper_dynamics import PaperDynamics, PAPER_DYNAMICS_VERSION

from .connectome import FILES, SOURCE, batches

DYNAMICS_VERSION = 'lif-sensory-histamine-v3'


def build_full(directory: Path, destination: Path | None = None):
    destination = destination or directory / 'full'
    if destination.exists():
        raise ValueError(f'{destination} already exists; choose a new output directory')
    annotations = feather.read_table(directory / 'body-annotations.feather')
    neurons = annotations.filter(pc.is_valid(annotations['superclass'])).sort_by('bodyId')
    ids = neurons['bodyId'].to_numpy()
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError('Expected unique annotated neuron IDs')
    values = pa.array(ids)
    n = len(ids)
    totals = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.full-build-', dir=destination.parent) as tmp:
        work = Path(tmp)
        with (work / 'edges.bin').open('wb') as out:
            for batch in batches(directory / 'connectome-weights.feather'):
                pre, post = batch.column('body_pre'), batch.column('body_post')
                counts = batch.column('weight')
                if pc.any(pc.less(counts, 0)).as_py():
                    raise ValueError('Negative anatomical synapse count')
                a, b = pc.is_in(pre, value_set=values), pc.is_in(post, value_set=values)
                kept = batch.filter(pc.and_(a, b))
                cross = batch.filter(pc.xor(a, b))
                totals['source_connection_rows'] += batch.num_rows
                totals['source_synapses'] += pc.sum(counts).as_py() or 0
                totals['retained_connection_rows'] += kept.num_rows
                totals['retained_synapses'] += pc.sum(kept.column('weight')).as_py() or 0
                totals['boundary_connection_rows'] += cross.num_rows
                totals['boundary_synapses'] += pc.sum(cross.column('weight')).as_py() or 0
                if kept.num_rows:
                    # Integer key preserves post/pre orientation and permits exact aggregation.
                    rows = np.searchsorted(ids, kept.column('body_post').to_numpy())
                    cols = np.searchsorted(ids, kept.column('body_pre').to_numpy())
                    np.column_stack((rows * n + cols, kept.column('weight').to_numpy())).astype(np.int64).tofile(out)
        raw = np.fromfile(work / 'edges.bin', dtype=np.int64).reshape(-1, 2)
        order = np.argsort(raw[:, 0], kind='stable')
        raw = raw[order]
        starts = np.r_[0, np.flatnonzero(np.diff(raw[:, 0])) + 1] if len(raw) else np.array([], dtype=int)
        keys = raw[starts, 0]
        counts = np.add.reduceat(raw[:, 1], starts) if len(raw) else np.array([], dtype=np.int64)
        rows, cols = keys // n, keys % n
        indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
        np.save(work / 'indptr.npy', indptr)
        np.save(work / 'indices.npy', cols.astype(np.int32))
        np.save(work / 'counts.npy', counts)
        np.save(work / 'body_ids.npy', ids)
        nt = feather.read_table(directory / 'body-neurotransmitters.feather')
        nt = {r['body']: r for r in nt.filter(pc.is_in(nt['body'], value_set=values)).to_pylist()}
        neurons = neurons.append_column('transmitter', pa.array([nt.get(int(i), {}).get('consensus_nt') for i in ids], type=pa.string()))
        neurons = neurons.append_column('transmitter_confidence', pa.array([nt.get(int(i), {}).get('predicted_nt_confidence') for i in ids], type=pa.float64()))
        feather.write_feather(neurons, work / 'neurons.feather')
        sources = []
        for local, remote in FILES.items():
            with (directory / local).open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            sources.append({'file': local, 'url': SOURCE + remote, 'sha256': digest})
        manifest = {'format': 'flylab-count-csr-v1', 'dataset': 'male-cns:v1.0', 'license': 'CC-BY',
                    'neurons': n, 'edges': len(counts), **dict(totals),
                    'excluded_internal_rows': totals['source_connection_rows'] - totals['retained_connection_rows'] - totals['boundary_connection_rows'],
                    'excluded_annotation_rows': annotations.num_rows - n,
                    'selection': 'Every body annotation with a non-null superclass; all induced edges, no top-k or weight threshold.',
                    'orientation': 'CSR rows=postsynaptic, columns=presynaptic; duplicate pairs sum exact int64 counts.',
                    'classes': dict(Counter(neurons['superclass'].to_pylist())), 'sources': sources}
        manifest['graph_sha256'] = hashlib.sha256(''.join(hashlib.sha256((work / f).read_bytes()).hexdigest() for f in ['body_ids.npy', 'indptr.npy', 'indices.npy', 'counts.npy']).encode()).hexdigest()
        (work / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (work / 'edges.bin').unlink()
        # Publish only a complete, validated artifact.
        assert int(counts.sum()) == totals['retained_synapses']
        work.rename(destination)
    return manifest


@dataclass(frozen=True)
class Physiology:
    """Versioned assumptions, not recordings; use paper() for current experiments.

    Constructor defaults retain the legacy checkpoint contract explicitly.
    """
    dt_ms: float = 1.
    membrane_ms: float = 20.
    synapse_ms: float = 5.
    refractory_ms: float = 2.
    gain: float = 8.
    glutamate_sign: float = -1.
    unknown_sign: float = 0.
    histamine_sign: float = -1.
    profile: str = 'legacy-normalized'
    delay_ms: float = 1.8

    @classmethod
    def paper(cls, **overrides):
        return cls(**{'dt_ms': .1, 'refractory_ms': 2.2, 'gain': .275,
                      'profile': 'shiu-2024', **overrides})

    def __post_init__(self):
        if self.profile not in {'legacy-normalized', 'shiu-2024'}:
            raise ValueError('Unknown physiology profile')
        if not all(math.isfinite(x) for k, x in asdict(self).items() if k != 'profile'):
            raise ValueError('Physiology values must be finite')
        if not (0 < self.dt_ms <= self.membrane_ms and self.dt_ms <= self.synapse_ms and self.refractory_ms >= 0 and 0 <= self.gain <= 100):
            raise ValueError('Invalid integration times or gain')
        if any(value not in {-1., 0., 1.} for value in (self.glutamate_sign, self.unknown_sign, self.histamine_sign)):
            raise ValueError('Transmitter signs must be -1, 0, or 1')
        if self.delay_ms < 0 or (self.profile == 'shiu-2024' and any(not math.isclose(round(v / self.dt_ms) * self.dt_ms, v) for v in [self.delay_ms, self.refractory_ms])):
            raise ValueError('Delay and refractory time must be whole integration ticks')


class FullBrain(PaperDynamics):
    @cached_property
    def neural_view(self):
        from .neural_view import NeuralView
        return NeuralView(self)

    @cached_property
    def anatomy(self):
        from .anatomy import AnatomyIndex
        return AnatomyIndex(self)

    def __init__(self, directory: Path, physiology: Physiology | None = None, device='cpu'):
        self.directory = directory
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        self.config = physiology or Physiology.paper()
        self.neurons = feather.read_table(directory / 'neurons.feather').to_pylist()
        self.ids = np.load(directory / 'body_ids.npy')
        self.lookup = {int(v): i for i, v in enumerate(self.ids)}
        ptr = np.load(directory / 'indptr.npy')
        indices = np.load(directory / 'indices.npy').astype(np.int64)
        counts = np.load(directory / 'counts.npy')
        n = len(self.ids)
        rows = np.repeat(np.arange(n), np.diff(ptr))
        total = np.bincount(rows, weights=counts, minlength=n).clip(1)
        signs = {'acetylcholine': 1., 'ach': 1., 'gaba': -1., 'glutamate': self.config.glutamate_sign, 'glut': self.config.glutamate_sign,
                 'histamine': self.config.histamine_sign}
        sign = np.array([signs.get(str(r['transmitter']).lower(), self.config.unknown_sign) for r in self.neurons])
        weights = counts * sign[indices] * self.config.gain
        if self.config.profile == 'legacy-normalized':
            weights /= total[rows]
        dtype = np.float64 if self.config.profile == 'shiu-2024' else np.float32
        self.weights = torch.sparse_csr_tensor(torch.from_numpy(ptr), torch.from_numpy(indices), torch.from_numpy(weights.astype(dtype)), size=(n, n), check_invariants=True)
        self.unresolved_neurons = int(sum(str(r['transmitter']).lower() not in signs for r in self.neurons))
        if self.config.profile == 'shiu-2024':
            self.prepare_paper()
        self.reset()
        if device != 'cpu':
            self.set_device(device)

    @property
    def dynamics_version(self):
        return PAPER_DYNAMICS_VERSION if self.config.profile == 'shiu-2024' else DYNAMICS_VERSION

    def reset(self):
        if self.config.profile == 'shiu-2024':
            self.reset_paper()
            return
        n = len(self.ids)
        self.voltage = torch.zeros(n)
        self.current = torch.zeros(n)
        self.refractory = torch.zeros(n)
        self.spikes = torch.zeros(n)
        self.output_gain = torch.ones(n)
        self.rates = torch.zeros(n)
        self.total_spikes = 0
        self.time_ms = 0.
        self.last_wall_seconds = 0.

    @torch.no_grad()
    def advance(self, drive: torch.Tensor, duration_ms=20., silence=None):
        if self.config.profile == 'shiu-2024':
            return self.advance_paper(drive, duration_ms, silence)
        if drive.shape != self.voltage.shape or not torch.isfinite(drive).all():
            raise ValueError('Drive must be a finite current for each neuron')
        steps = round(duration_ms / self.config.dt_ms)
        if steps < 1 or not math.isclose(steps * self.config.dt_ms, duration_ms):
            raise ValueError('Duration must be a positive multiple of integration time')
        started = time.perf_counter()
        c = self.config
        rate_decay = math.exp(-c.dt_ms / 50.)
        count = torch.zeros_like(self.voltage)
        for _ in range(steps):
            self.current *= math.exp(-c.dt_ms / c.synapse_ms)
            if torch.any(self.spikes):
                self.current += torch.mv(self.weights, self.spikes * self.output_gain)
            self.refractory = (self.refractory - c.dt_ms).clamp_min(0)
            self.voltage += (1 - math.exp(-c.dt_ms / c.membrane_ms)) * (-self.voltage + self.current + drive)
            self.voltage[self.refractory > 0] = 0
            if silence is not None:
                self.voltage[silence] = 0
            self.spikes = (self.voltage >= 1.).float()
            fired = self.spikes.bool()
            self.voltage[fired] = 0
            self.refractory[fired] = c.refractory_ms
            count += self.spikes
            # Filter each integration step so API batching cannot change motor drive.
            self.rates.mul_(rate_decay).add_(self.spikes, alpha=(1 - rate_decay) * 1000. / c.dt_ms)
        self.total_spikes += int(count.sum())
        self.time_ms += duration_ms
        self.last_wall_seconds = time.perf_counter() - started
        return self.rates

    def summary(self):
        return {**self.manifest, 'physiology': asdict(self.config), 'dynamics_version': self.dynamics_version, 'unresolved_transmitters': self.unresolved_neurons,
                'device': self.voltage.device.type, 'precision': str(self.voltage.dtype),
                'execution': self.execution_summary() if self.config.profile == 'shiu-2024' else None,
                'assumptions': ('Shiu 2024 timing and raw count efficacy transferred to MaleCNS, not a reproduction on the original FlyWire specimen. ' if self.config.profile == 'shiu-2024' else 'Legacy normalized LIF. ') + 'Physiology is assumed, including spiking approximation of graded visual receptors. ACh positive; GABA, histamine and default glutamate negative. Unresolved/modulatory transmitters contribute zero fast current because receptor effects are unknown, not for acceleration. Their neurons, state and anatomical edges remain present.',
                'total_spikes': self.total_spikes, 'active_neurons': int((self.rates > 1).sum()), 'neural_time_ms': self.time_ms,
                'last_neural_wall_seconds': self.last_wall_seconds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_full(args.data, args.output), indent=2))
