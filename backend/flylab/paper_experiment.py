"""Untrained feeding experiment on the paper's complete FlyWire v630 graph.

This is separate from MaleCNS: IDs, sex, coverage and transmitter conventions
must never be interchanged. No muscle animation is inferred from MN9 firing.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import threading
import copy

import numpy as np
import pyarrow.parquet as pq
import torch

from .full_connectome import Physiology
from .paper_dynamics import PaperDynamics, PAPER_DYNAMICS_VERSION

COMMIT = '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960'
SOURCES = {
    '2023_03_23_completeness_630_final.csv': 'e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb',
    '2023_03_23_connectivity_630_final.parquet': '94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3',
}
PROTOCOL = json.loads((Path(__file__).parent / 'assets/shiu2024_protocol.json').read_text())


class PaperBrain(PaperDynamics):
    def __init__(self, directory: Path):
        for name, digest in SOURCES.items():
            with (directory / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'Paper source checksum mismatch: {name}')
        with (directory / next(iter(SOURCES))).open() as stream:
            rows = csv.reader(stream)
            next(rows)
            self.ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
        self.lookup = {int(i): j for j, i in enumerate(self.ids)}
        columns = ['Presynaptic_Index', 'Postsynaptic_Index', 'Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity', 'Excitatory x Connectivity']
        table = pq.read_table(directory / list(SOURCES)[1], columns=columns)
        pre, post, pre_id, post_id, counts, signed = (table[c].to_numpy() for c in columns)
        if not np.array_equal(self.ids[pre], pre_id) or not np.array_equal(self.ids[post], post_id):
            raise ValueError('Paper neuron ordering disagrees with connectivity')
        if (counts < 0).any() or not np.array_equal(np.abs(signed), counts):
            raise ValueError('Invalid paper signed synapse counts')
        if not np.all(pre[1:] >= pre[:-1]):
            raise ValueError('Expected pinned presynaptic ordering')
        self.config = Physiology.paper()
        self.dtype = torch.float64
        self.delay_steps = round(self.config.delay_ms / self.config.dt_ms)
        self.out_ptr = torch.from_numpy(np.r_[0, np.cumsum(np.bincount(pre, minlength=len(self.ids)))])
        self.out_post = torch.tensor(post, dtype=torch.int64)
        self.out_weight = torch.tensor(signed, dtype=torch.float64) * self.config.gain
        self.manifest = {'dataset': 'FlyWire female v630 (Shiu 2024)', 'neurons': len(self.ids),
                         'connection_rows': len(pre), 'synapses': int(counts.sum()),
                         'source_commit': COMMIT, 'source_sha256': SOURCES,
                         'dynamics_version': PAPER_DYNAMICS_VERSION,
                         'selection': 'Every source neuron and connection row; no pruning or weight threshold.',
                         'transmitters': 'Uses published signed counts verbatim, including the paper convention for modulatory transmitters.'}
        self.reset_paper()


def feeding_experiment(directory, duration_ms=1000., seed=42, progress=None, stop=None):
    """One seeded trial per condition; NOT the paper's 30-trial validation."""
    if duration_ms <= 0 or not float(duration_ms).is_integer() or duration_ms % 20:
        raise ValueError('Use a positive multiple of 20 ms')
    brain = PaperBrain(Path(directory))
    sugar = [brain.lookup[i] for i in PROTOCOL['neu_sugar']]
    bitter = [brain.lookup[i] for i in PROTOCOL['neu_bitter']]
    outputs = [brain.lookup[i] for i in PROTOCOL['mn9']]
    results = []
    # Independent receptor RNG streams are not shared between conditions, so this
    # is a reproducible seed, not an assertion of matched stochastic inputs.
    for name, sugar_hz, bitter_hz in [('no-input', 0, 0), ('sugar', 100, 0), ('sugar+bitter', 100, 100)]:
        brain.reset_paper(seed)
        brain.refractory_ticks[sugar + bitter] = 0
        rate = torch.zeros(len(brain.ids), dtype=torch.float64)
        rate[sugar], rate[bitter] = sugar_hz, bitter_hz
        started = time.perf_counter()
        for step in range(round(duration_ms / 20)):
            if stop is not None and stop.is_set():
                return {**brain.manifest, 'cancelled': True, 'results': results}
            brain.advance_paper(torch.zeros_like(rate), 20, poisson_hz=rate)
            if progress:
                progress({'condition': name, 'simulated_ms': brain.time_ms, 'target_ms': duration_ms,
                          'neurons': len(brain.ids), 'spikes': brain.total_spikes})
        results.append({'condition': name, 'sugar_hz': sugar_hz, 'bitter_hz': bitter_hz,
                        'mn9_hz': (brain.spike_counts[outputs].double() * 1000 / duration_ms).tolist(),
                        'total_spikes': brain.total_spikes, 'wall_seconds': time.perf_counter() - started})
    return {**brain.manifest, 'seed': seed, 'duration_ms': duration_ms, 'trials_per_condition': 1,
            'results': results, 'trained_parameters': 0,
            'limits': 'A single-seed protocol smoke test, not a replication of all published predictions or trial statistics. MN9 spikes are not validated muscle motion. MaleCNS is a different specimen.'}


class PaperExperimentRunner:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status = {'running': False, 'result': None, 'error': None, 'progress': None}
        saved = directory / 'feeding-42-1000ms.json'
        if saved.exists():
            result = json.loads(saved.read_text())
            if result.get('dynamics_version') == PAPER_DYNAMICS_VERSION and result.get('source_sha256') == SOURCES:
                self.status['result'] = result

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def start(self, duration_ms=1000, seed=42):
        with self.lock:
            if self.status['running']:
                raise ValueError('A paper experiment is already running')
            if not all((self.directory / name).exists() for name in SOURCES):
                raise ValueError('Download the pinned paper data with scripts/reproduce_paper.py --download')
            self.stop_event.clear()
            self.status.update(running=True, error=None, progress=None)
        def progress(value):
            with self.lock:
                self.status['progress'] = value
        def work():
            try:
                result = feeding_experiment(self.directory, duration_ms, seed, progress, self.stop_event)
                with self.lock:
                    self.status['result'] = result
            except Exception as exc:
                with self.lock:
                    self.status['error'] = str(exc)
            finally:
                with self.lock:
                    self.status['running'] = False
        threading.Thread(target=work, name='paper-experiment', daemon=True).start()
