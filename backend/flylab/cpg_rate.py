"""Offline full-graph rate experiment; NOT the live Shiu LIF physiology.

Independent implementation of the rectified-tanh rate equation and parameter
distributions described by Pugliese et al., doi:10.1101/2025.09.12.675944.
The whole-CNS extension and VNC-wide size normalization require validation.
No pruning, synthetic oscillator, gait reference or trained decoder is used.
"""
import numpy as np
import torch

VERSION = 'whole-cns-rate-experiment-v1'


class RateExperiment:
    def __init__(self, brain, volumes, *, device='cpu', seed=1, dt_ms=.5, volume_normalizer=None):
        n = len(brain.ids)
        volumes = np.asarray(volumes, dtype=float)
        if volumes.shape != (n,) or np.isinf(volumes).any() or np.any(np.isfinite(volumes) & (volumes <= 0)):
            raise ValueError('Expected one positive measured volume or NaN per neuron')
        if device not in {'cpu', 'mps'} or not 0 < dt_ms <= 1:
            raise ValueError('Rate experiment requires CPU/MPS and dt in (0, 1] ms')
        vnc = np.array([str(r.get('superclass', '')).startswith('vnc_') for r in brain.neurons])
        valid = np.isfinite(volumes)
        if not (valid & vnc).any():
            raise ValueError('Measured VNC volumes required for normalization')
        normalizer = float(np.median(volumes[valid & vnc])) if volume_normalizer is None else float(volume_normalizer)
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError('Volume normalizer must be positive and finite')
        size = np.where(valid, volumes, normalizer) / normalizer
        rng = np.random.default_rng(seed)
        def positive(mean, sd):
            values = rng.normal(mean, sd, n)
            while np.any(values <= 0):
                bad = values <= 0; values[bad] = rng.normal(mean, sd, bad.sum())
            return values
        def tensor(values):
            return torch.tensor(values, dtype=torch.float32, device=device)
        self.tau = tensor(positive(20., 2.))
        self.slope = tensor(positive(1., .1) / size)
        self.threshold = tensor(positive(7.5, .6) * size)
        self.cap = tensor(positive(200., 10.))
        if dt_ms >= float(self.tau.min()):
            raise ValueError('Timestep exceeds smallest membrane time constant')
        # Reuse the identical CSR structure but construct efficacy from integer
        # source counts, independent of paper-profile gain or learned weights.
        signs = {'acetylcholine': 1., 'gaba': -1., 'glutamate': -1., 'histamine': -1.}
        nt = np.array([signs.get(str(r.get('transmitter')).lower(), 0.) for r in brain.neurons])
        columns = brain.weights.col_indices().cpu()
        counts = np.load(brain.directory / 'counts.npy')
        values = torch.tensor(counts * nt[columns.numpy()] * .03, dtype=torch.float32)
        self.weights = torch.sparse_csr_tensor(brain.weights.crow_indices().cpu(), columns, values, size=(n,n), check_invariants=True)
        self.metal = None
        if device == 'mps':
            from .metal_dynamics import MetalDynamics
            self.metal = MetalDynamics(self.weights)
        self.rates = torch.zeros(n, dtype=torch.float32, device=device)
        self.dt_ms, self.ticks = dt_ms, 0
        self.metadata = {'version': VERSION, 'precision': 'float32', 'device': device,
                         'neurons': n, 'edges': self.weights._nnz(), 'seed': seed,
                         'graph_sha256': brain.manifest['graph_sha256'],
                         'dt_ms': dt_ms, 'integrator': 'Heun RK2',
                         'synaptic_count_scale': .03, 'transmitter_signs': signs,
                         'unmodeled_transmitter_effects': int(np.count_nonzero(nt == 0)),
                         'volume_normalizer': normalizer, 'volume_missing': int((~valid).sum()),
                         'normalization': ('Median measured volume across all vnc_ neurons.' if volume_normalizer is None else 'Explicit reference volume supplied by experiment; no graph selection change.') + ' Missing volumes use the normalizer.',
                         'scope': 'Unvalidated full-CNS extension; source study uses different graph selections and an adaptive solver.'}

    def derivative(self, rates, drive):
        incoming = self.metal.deliver(rates) if self.metal else torch.mv(self.weights, rates)
        target = self.cap * torch.tanh(self.slope / self.cap * (incoming + drive - self.threshold))
        return (target.clamp_min(0) - rates) / self.tau

    @torch.no_grad()
    def advance(self, drive, duration_ms):
        ticks = round(duration_ms / self.dt_ms)
        if ticks < 1 or abs(ticks*self.dt_ms-duration_ms) > 1e-8:
            raise ValueError('Rate duration must be an integral number of neural steps')
        if drive.shape != self.rates.shape or not torch.isfinite(drive).all():
            raise ValueError('Expected finite input to every neuron')
        drive = drive.to(self.rates)
        for _ in range(ticks):
            first = self.derivative(self.rates, drive)
            second = self.derivative(self.rates + self.dt_ms * first, drive)
            self.rates.add_((first + second) * (.5 * self.dt_ms))
        self.ticks += ticks
        if not torch.isfinite(self.rates).all() or (self.rates < -1e-5).any():
            raise RuntimeError('Unstable rate experiment; no clipping or silent repair')
        return self.rates
