"""Targeted descending command, never a muscle policy or gait generator.

DNg100 and DNb08: Pugliese et al., doi:10.1101/2025.09.12.675944
(preprint). Tonic excitation and resulting body behavior are experimental.
"""
import math
import torch

PROFILE = 'descending-walk-v1'
TARGETS = ('DNg100', 'DNb08')
CORE = ('IN17A001', 'INXXX466', 'IN16B036', 'IN19A007')


class LocomotorCommand:
    def __init__(self, brain):
        self.brain = brain
        self.groups = {name: torch.tensor([i for i, row in enumerate(brain.neurons)
                      if row.get('type') == name], dtype=torch.long) for name in (*TARGETS, *CORE)}
        self.reset()

    def reset(self):
        self.target, self.amplitude, self.expires = 'DNg100', 0., 0.

    def submit(self, target, amplitude, duration, now):
        if target not in TARGETS or not len(self.groups[target]):
            raise ValueError('Identified descending walking neurons are unavailable')
        if (not all(math.isfinite(v) for v in (amplitude, duration, now))
                or not 0 <= amplitude <= 3 or not .02 <= duration <= 5 or now < 0):
            raise ValueError('Invalid descending walking command')
        self.target, self.amplitude, self.expires = target, float(amplitude), now + duration

    def apply(self, drive, now):
        if now < self.expires:
            # Same threshold-relative units as regional stimulation. This
            # modifies only the named DNs; no intermediate or motor drive.
            drive[self.groups[self.target]] += self.amplitude

    def state_dict(self):
        return {'profile': PROFILE, 'target': self.target, 'amplitude': self.amplitude, 'expires': self.expires}

    def load_state_dict(self, state):
        if set(state) != {'profile', 'target', 'amplitude', 'expires'} or state['profile'] != PROFILE:
            raise ValueError('Walking command profile mismatch')
        a, end, target = state['amplitude'], state['expires'], state['target']
        if target not in TARGETS or not math.isfinite(a) or not 0 <= a <= 3 or not math.isfinite(end) or end < 0:
            raise ValueError('Invalid saved walking command')
        if a and not len(self.groups[target]):
            raise ValueError('Saved walking neurons absent from graph')
        self.target, self.amplitude, self.expires = target, a, end

    def summary(self, now):
        rates = self.brain.rates.detach().cpu()
        return {**self.state_dict(), 'remaining_s': max(0., self.expires - now) if self.amplitude else 0.,
                'targets': [{'type': name, 'body_ids': [int(self.brain.ids[i]) for i in self.groups[name]],
                             'mean_hz': float(rates[self.groups[name]].float().mean()) if len(self.groups[name]) else 0.}
                            for name in TARGETS],
                'core': [{'type': name, 'neurons': len(self.groups[name]),
                          'mean_hz': float(rates[self.groups[name]].float().mean()) if len(self.groups[name]) else 0.}
                         for name in CORE]}
