"""Annotation-backed peripheral channels with explicit assumed mechanical routing.

Never assign unidentified MNs by nearest position or arbitrary sorted order.
The source type identifies a muscle group; axis, polarity, rate-to-force gain,
and sensory tuning remain engineering hypotheses.
"""
from __future__ import annotations

import numpy as np
import torch
from .body import LEGS, ACTIVE_DOF
from .full_connectome import FullBrain
from .senses import SensorSuite

from .motor_mapping import resolve_motor, inventory_summary, MAPPING_PROFILE, LEGACY_PROFILE, PROFILES, PRETARSAL_PROFILE, PERIPHERAL_PROFILE
from .peripheral_mechanics import CHANNEL_COUNT, MUSCLES

REGION_NAMES = ['optic', 'antennal', 'mushroom', 'descending', 'vnc', 'motor']


def region_for(row):
    superclass = row['superclass'] or ''
    if 'motor' in superclass:
        return 'motor'
    if superclass.startswith('ol_') or superclass.startswith('visual'):
        return 'optic'
    if superclass.startswith('descending'):
        return 'descending'
    if superclass.startswith('vnc_') or superclass.startswith('ascending'):
        return 'vnc'
    if row.get('class') == 'olfactory' or str(row.get('type', '')).startswith(('ORN_', 'AL')):
        return 'antennal'
    if str(row.get('type', '')).startswith(('KC', 'MBON', 'DAN', 'PAM', 'PPL')):
        return 'mushroom'
    return None  # Do not label the rest of the central brain as mushroom body.


class NeuromuscularBridge:
    def __init__(self, brain: FullBrain, mapping_profile=MAPPING_PROFILE):
        if mapping_profile not in PROFILES:
            raise ValueError("Unknown muscle mapping profile")
        self.mapping_profile = mapping_profile
        self.brain = brain
        self._retina = None
        self.regions = {r: torch.tensor([i for i, n in enumerate(brain.neurons) if region_for(n) == r], dtype=torch.long) for r in REGION_NAMES}
        self.olfactory = [[], []]
        self.visual = [[], []]
        self.auditory = [[], []]
        self.wind_gravity = [[], []]
        self.touch = [[] for _ in LEGS]
        self.position = [[] for _ in LEGS]
        self.mapping = []
        count = CHANNEL_COUNT if mapping_profile == PERIPHERAL_PROFILE else 90 if mapping_profile == PRETARSAL_PROFILE else 84
        self.channels = [[] for _ in range(count)]
        self.unmapped = []
        self.motor_inventory = []
        self.channel_weights = [[] for _ in range(count)]
        for i, row in enumerate(brain.neurons):
            side = row.get('rootSide')
            if side in {'L', 'R'}:
                if row['superclass'] == 'ol_sensory' and row.get('class') == 'visual':
                    self.visual[side == 'R'].append(i)
                if row['superclass'] == 'cb_sensory' and str(row.get('type', '')).startswith('JO-'):
                    if row.get('subclass') == 'auditory':
                        self.auditory[side == 'R'].append(i)
                    elif row.get('subclass') == 'wind_gravity':
                        self.wind_gravity[side == 'R'].append(i)
            if row.get('class') == 'olfactory' and side in {'L', 'R'}:
                self.olfactory[side == 'R'].append(i)
            if row['superclass'] == 'vnc_sensory' and side in {'L', 'R'}:
                pair = {'ProLN': 'F', 'MesoLN': 'M', 'MetaLN': 'H'}.get(row.get('entryNerve'))
                if pair:
                    leg = LEGS.index(side + pair)
                    if row.get('class') == 'mechanosensory_tactile':
                        self.touch[leg].append(i)
                    elif row.get('class') == 'mechanosensory_proprioceptive':
                        self.position[leg].append(i)
            if row['superclass'] not in {'vnc_motor', 'cb_motor'}:
                continue
            record = resolve_motor(row, mapping_profile)
            self.motor_inventory.append(record)
            if record['status'] != 'mapped':
                self.unmapped.append(record)
                continue
            leg = record['leg']
            transmissions = []
            if 'peripheral_channel' in record:
                channel = record['peripheral_channel']
                self.channels[channel].append(i)
                self.channel_weights[channel].append(1.)
                muscle = MUSCLES[channel-90]
                transmissions = [{'joint': name, 'actuator': channel, 'coefficient': abs(c),
                                  'torque_sign': 1 if c < 0 else -1} for name, c in muscle.joints]
            for link, axis, polarity, coefficient in record['projections']:
                channel = 84 + LEGS.index(leg) if link == 'pretarsus' else (LEGS.index(leg) * 7 + ACTIVE_DOF.index((link, axis))) * 2 + polarity
                self.channels[channel].append(i)
                self.channel_weights[channel].append(coefficient)
                transmissions.append({'joint': f'{leg.lower()}_{link}_{axis}', 'actuator': channel,
                                      'coefficient': coefficient, 'torque_sign': 1 if polarity == 0 else -1})
            record.update(transmissions=transmissions, joint=', '.join(t['joint'] for t in transmissions),
                          actuator=transmissions[0]['actuator'])
            self.mapping.append(record)
        self.olfactory = [torch.tensor(x, dtype=torch.long) for x in self.olfactory]
        self.visual = [torch.tensor(x, dtype=torch.long) for x in self.visual]
        self.auditory = [torch.tensor(x, dtype=torch.long) for x in self.auditory]
        self.wind_gravity = [torch.tensor(x, dtype=torch.long) for x in self.wind_gravity]
        self.touch = [torch.tensor(x, dtype=torch.long) for x in self.touch]
        self.position = [torch.tensor(x, dtype=torch.long) for x in self.position]
        self.channels = [torch.tensor(x, dtype=torch.long) for x in self.channels]
        self.channel_weights = [torch.tensor(x, dtype=torch.float64) for x in self.channel_weights]
        # Compact parameters alter regional neural efficacy, sensory gain and motor
        # rate gain. They never introduce new neuron-neuron edges or muscle channels.
        self.parameters = np.zeros(8, dtype=np.float32)
        self.set_parameters(self.parameters)

    def retina(self):
        if self._retina is None:
            from .retina import RetinalRouting
            self._retina = RetinalRouting(self.brain)
        return self._retina

    def set_parameters(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (8,) or not np.isfinite(values).all() or np.max(np.abs(values)) > 2:
            raise ValueError('Expected eight finite log gains in [-2, 2]')
        self.parameters = values.copy()
        self.brain.output_gain = torch.ones_like(self.brain.voltage)
        for i, name in enumerate(REGION_NAMES):
            self.brain.output_gain[self.regions[name]] = float(np.exp(values[i]))

    def sensory_drive(self, body, source, intensity=.7, spatial=True, frame=None):
        drive = torch.zeros(len(self.brain.ids))
        frame = frame or SensorSuite().sample(body, source, intensity, spatial)
        gain = float(np.exp(self.parameters[6]))
        for i in range(2):
            drive[self.olfactory[i]] = float(frame['odor'][i] * 3 * gain)
            # Side is measured; pooling all pixels is explicit until retinotopic
            # eye-column assignments are imported. Never assign pixels by ID order.
            if frame['vision'].get('model', 'legacy-grid-v2') == 'legacy-grid-v2':
                drive[self.visual[i]] = float(frame['vision']['mean'][i] * 3 * gain)
            drive[self.auditory[i]] = float(frame['hearing'][i] * 3 * gain)
            drive[self.wind_gravity[i]] = float(frame['wind'][i] * 3 * gain)
        if frame['vision'].get('model') == 'compound-retina-v1':
            self.retina().apply(drive, frame['vision'], 3 * gain)
        for leg in range(6):
            drive[self.touch[leg]] = float(frame['touch'][leg] * 3 * gain)
            # Generic position tuning is declared, not identified receptor tuning.
            drive[self.position[leg]] = float(frame['proprioception'][leg] * 2 * gain)
        return drive

    def muscles(self):
        action = np.zeros(len(self.channels), dtype=np.float32)
        gain = np.exp(self.parameters[7])
        rates = self.brain.rates.detach().cpu()
        for channel, indices in enumerate(self.channels):
            if len(indices):
                rate = rates[indices].mean() if self.mapping_profile == LEGACY_PROFILE else (rates[indices].double() * self.channel_weights[channel]).mean()
                action[channel] = np.clip(float(rate) / 100 * gain, 0, 1)
        return action

    def command(self, body, source, intensity=.7, spatial=True, pulses=None, silenced=(), frame=None):
        drive = self.sensory_drive(body, source, intensity, spatial, frame)
        for region, amplitude in (pulses or {}).items():
            drive[self.regions[region]] += amplitude
        if self.brain.config.profile == 'shiu-2024':
            # Workbench controls are threshold-relative; the paper state uses mV
            # with a 7 mV rest-to-threshold gap. This is assumed sensory encoding,
            # not the Poisson GRN protocol used in the paper reproduction.
            drive *= 7.
        mask = torch.zeros(len(drive), dtype=torch.bool)
        for region in silenced:
            mask[self.regions[region]] = True
        self.brain.advance(drive, silence=mask)
        action = self.muscles()
        if 'motor' in silenced:
            action[:] = 0
        return action

    def step(self, body, source, intensity=.7, spatial=True, pulses=None, silenced=(), frame=None):
        action = self.command(body, source, intensity, spatial, pulses, silenced, frame)
        body.step_muscles(action)
        return action

    def summary(self):
        return {**inventory_summary(self.motor_inventory), 'mapping_profile': self.mapping_profile, 'available_profiles': list(PROFILES),
                'mapped_motor_neurons': len(self.mapping), 'unmapped_motor_neurons': len(self.unmapped),
                'actuated_channels': sum(bool(len(x)) for x in self.channels), 'total_channels': len(self.channels),
                'sensory_neurons': {'olfactory': sum(map(len, self.olfactory)), 'tactile': sum(map(len, self.touch)), 'proprioceptive': sum(map(len, self.position)),
                                    'visual': sum(map(len, self.visual)), 'auditory': sum(map(len, self.auditory)), 'wind_gravity': sum(map(len, self.wind_gravity))},
                'regional_neuron_counts': {k: len(v) for k, v in self.regions.items()},
                'unassigned_region_neurons': len(self.brain.ids) - sum(map(len, self.regions.values())),
                'mapping': self.mapping, 'unmapped': self.unmapped,
                'parameters': self.parameters.tolist(),
                'source': 'https://male-cns.janelia.org/download/',
                'limitations': 'Muscle-group identities and sensory classes/sides use dataset annotations. Leg and positioning-mouthpart projections use published primary actions. Peripheral wing, neck and haltere transmissions are uncalibrated hypotheses; internal wall elements do not simulate fluid transport. Hill actuators do not implement asynchronous flight power or aerodynamics; ipsilateral assignment, fixed moment arms and force scaling remain assumptions. Coxa remotor/abductor drive is split equally between two axes, without calibrated muscle geometry. Legacy vision pools brightness per eye. Compound vision uses inferred columns and an unvalidated cross-specimen optical registration; UV and polarization are absent. Auditory/wind inputs use annotated JO subclasses with assumed tuning. Odor A/B share generic olfactory tuning. Unmapped actuators receive zero excitation. No gait tracker or target-bearing input in this mode.'}
