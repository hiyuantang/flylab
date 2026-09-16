"""Additional source-backed motor routes, with explicitly approximate mechanics.

v4's actuator ordering and mechanics remain unchanged. New transmissions are
appended in v5; old checkpoints keep their original physical model.
"""
import xml.etree.ElementTree as ET

from .peripheral_mechanics import Muscle, MUSCLES as V4_MUSCLES, add_mechanics as add_v4

PROFILE = 'muscle-routing-v5'
MODEL = 'peripheral-v2'
NECK_SOURCE = 'https://www.nature.com/articles/s41586-024-07222-5'
MANC_SOURCE = 'https://elifesciences.org/articles/96084'


def additional_muscles():
    result = []
    for side, sign in [('L', 1), ('R', -1)]:
        s = side.lower()
        # Gorko 2024 Fig. 4h: CvN3--8 innervate VL1. Only the four
        # explicitly named types in MaleCNS are routed; no GNG aliases guessed.
        result.append(Muscle(f'{s}_vl1', 'CvN4', 'nm', side, 'VL1',
            (('c_head_pitch', -.025),), 20, NECK_SOURCE,
            'Fig. 4h identifies VL1; CvN4--7 rates are pooled per side. Positive pitch about +Y lowers the head. Pitch-only transmission, force and elasticity are uncalibrated; cervical sclerites and proprioceptive convergence are not reconstructed.'))
        # Cheong Supplementary file 3 gives FNM2 -> AD, certainty 2/5,
        # specifically based on its cross-midline axon. Do not apply the
        # same-side rule used for the other peripheral neurons.
        target_side = 'r' if side == 'L' else 'l'
        result.append(Muscle(f'{target_side}_ad', 'FNM2', 'nm', side, 'AD',
            (('c_head_turn', .025 * sign),), 20, MANC_SOURCE,
            'Tentative FNM2 -> AD match (source score 2/5); axon crosses the midline, so the modeled adductor is contralateral to the soma. Yaw toward that side is an uncalibrated effective cervical transmission, not measured insertion geometry.'))
        result.append(Muscle(f'{s}_iii4', 'MNwm35', 'wm', side, 'iii4',
            ((f'{s}_wing_spread', .025), (f'{s}_wing_feather', -.0075)),
            20, MANC_SOURCE,
            'Tentative iii4 match (source score 1/5; driver labels multiple motor neurons). Shares the existing third-axillary effective projection; distinct sclerite mechanics and phase-dependent steering are unvalidated.'))
        # The paper identifies this muscle family, but cannot distinguish
        # hb1 from hb2. Preserve the ambiguity in each actuator name/target.
        for neuron in ['MNhm42', 'MNhm43']:
            result.append(Muscle(f'{s}_hb1_or_hb2_{neuron.lower()}', neuron, 'hm', side,
                'hb1 or hb2', ((f'{s}_haltere_trim', -.015),), 10, MANC_SOURCE,
                'Supplementary file 3 identifies either hb1 or hb2; exact target is unresolved. Each neuron class retains a separate force channel with an assumed trim projection. This is a muscle-family approximation, not an exact hb1/hb2 assignment.'))
    return tuple(result)


ADDITIONAL_MUSCLES = additional_muscles()
MUSCLES = V4_MUSCLES + ADDITIONAL_MUSCLES
CHANNEL_COUNT = 90 + len(MUSCLES)
LOOKUP = {(m.neuron_type, m.subclass, m.side): 90+i for i, m in enumerate(MUSCLES)}
for side in ['L', 'R']:
    for neuron in ['CvN5', 'CvN6', 'CvN7']:
        LOOKUP[(neuron, 'nm', side)] = LOOKUP[('CvN4', 'nm', side)]


def add_mechanics(root, bodies, parameters):
    add_v4(root, bodies, parameters)
    ET.SubElement(bodies['c_head'], 'joint', name='c_head_pitch', type='hinge',
                  axis='0 1 0', range='-.35 .35', stiffness='2', damping='.01',
                  armature='1e-6', springref='0')
    tendons, actuators = root.find('tendon'), root.find('actuator')
    for muscle in ADDITIONAL_MUSCLES:
        tendon = ET.SubElement(tendons, 'fixed', name=muscle.name+'_transmission')
        for name, coefficient in muscle.joints:
            ET.SubElement(tendon, 'joint', joint=name, coef=str(coefficient))
        ET.SubElement(actuators, 'muscle', name=muscle.name,
                      tendon=muscle.name+'_transmission', lengthrange='-.15 .15',
                      force=str(muscle.force*parameters.strength_scale),
                      timeconst='.002 .004', fpmax='.00001')
