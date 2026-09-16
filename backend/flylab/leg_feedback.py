"""Versioned FeCO feedback; identities are measured, tuning is a hypothesis.

Opposing motor effects: https://elifesciences.org/reviewed-preprints/97766v1,
Figure 59. The stabilizing direction assignment below is inferred from those
effects, not a measurement of individual MaleCNS receptor tuning.
"""
from collections import Counter
import numpy as np
import torch
from .body import LEGS

PROFILE = 'feco-opponent-v1'
LEG_NERVES = {'ProLN': 'F', 'VProN': 'F', 'DProN': 'F', 'ProAN': 'F',
              'MesoLN': 'M', 'MetaLN': 'H'}
TUNING = {'SNpp50': 'flexed', 'SNpp51': 'extended',
          'SNpp39': 'extending', 'SNpp41': 'flexing'}


def sample_legs(body, enabled=True):
    """Actual femur-tibia opening and its hinge-derived angular velocity.

    Independent of camera orientation and mirrored mesh conventions. No state
    is advanced or perturbed to obtain these measurements.
    """
    d = body.data
    opening, velocity = [], []
    for leg in LEGS:
        p, c, e = [d.xpos[body.body_ids[f'{leg.lower()}_{part}']]
                   for part in ('trochanterfemur', 'tibia', 'tarsus1')]
        u, v = p - c, e - c
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        cosine = float(np.clip(u @ v, -1, 1))
        theta = float(np.arccos(cosine))
        jid = body.model.joint(f'{leg.lower()}_tibia_pitch').id
        derivative = -float(u @ np.cross(d.xaxis[jid], v)) / max(np.sqrt(1 - cosine*cosine), 1e-8)
        opening.append(theta)
        velocity.append(float(derivative * d.qvel[body.model.jnt_dofadr[jid]]))
    angle, speed = np.asarray(opening), np.asarray(velocity)
    return {'profile': PROFILE, 'opening_rad': opening, 'velocity_rad_s': velocity,
            'signals': {'flexed': ((1 - angle / np.pi) * enabled).tolist(),
                        'extended': (angle / np.pi * enabled).tolist(),
                        'extending': (np.clip(speed / 20., 0, 1) * enabled).tolist(),
                        'flexing': (np.clip(-speed / 20., 0, 1) * enabled).tolist()}}


class LegFeedback:
    def __init__(self, neurons):
        self.groups = {(leg, channel): [] for leg in range(6) for channel in TUNING.values()}
        self.unresolved = 0
        for i, row in enumerate(neurons):
            if row.get('superclass') != 'vnc_sensory' or row.get('class') != 'mechanosensory_proprioceptive':
                continue
            side, pair = row.get('rootSide'), LEG_NERVES.get(row.get('entryNerve'))
            channel = TUNING.get(row.get('type'))
            if side in {'L', 'R'} and pair and channel:
                self.groups[LEGS.index(side + pair), channel].append(i)
            else:
                self.unresolved += 1
        self.groups = {key: torch.tensor(value, dtype=torch.long) for key, value in self.groups.items()}

    def apply(self, drive, frame, gain):
        if frame.get('profile') != PROFILE:
            raise ValueError('Leg feedback profile mismatch')
        for (leg, channel), indices in self.groups.items():
            drive[indices] = float(frame['signals'][channel][leg] * 2 * gain)

    def summary(self):
        counts = Counter()
        for (_, channel), indices in self.groups.items():
            counts[channel] += len(indices)
        return {'profile': PROFILE, 'neurons': sum(counts.values()), 'channels': dict(counts),
                'unresolved_proprioceptors': self.unresolved,
                'assumptions': 'Opposing FeCO tuning inferred from motor effects; angle curves and 20 rad/s velocity scale uncalibrated. Club vibration and unresolved receptor tuning receive no invented external drive. All neurons and connections remain in the brain.'}
