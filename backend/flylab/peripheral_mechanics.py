"""Versioned effective mechanics for identified non-leg muscle targets.

Identity is annotation-backed; transmission coefficients, hinge coordinates,
force and elasticity are engineering assumptions. Wing/haltere power uses a
quasi-static Hill approximation, NOT asynchronous stretch-activated flight.
No unidentified motor type is assigned an actuator here.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

PROFILE = 'muscle-routing-v4'
MODEL = 'peripheral-v1'
MOUTH_SOURCE = 'https://elifesciences.org/articles/54978'
MOTOR_SOURCE = 'https://elifesciences.org/articles/96084'
NECK_SOURCE = 'https://www.nature.com/articles/s41586-025-08925-z'


@dataclass(frozen=True)
class Muscle:
    name: str
    neuron_type: str
    subclass: str
    side: str
    target: str
    joints: tuple[tuple[str, float], ...]
    force: float
    source: str
    interpretation: str


def muscle_specs():
    specs = []
    def add(neuron, subclass, side, target, joints, force, source, note):
        specs.append(Muscle(f'{side.lower()}_{target}', neuron, subclass, side, target,
                            tuple(joints), force, source, note))
    for side, sign in [('L', 1), ('R', -1)]:
        s = side.lower()
        # Table 3: primary positioning action, without inventing distant-joint
        # coupling observed in activation experiments.
        for neuron, target, joint, torque in [
            ('MN1', 'm1', 'c_rostrum_protraction', -1),
            ('MN2Da', 'm2Da', 'c_rostrum_protraction', -1),
            ('MN2Db', 'm2Db', 'c_rostrum_protraction', -1),
            ('MN2V', 'm2V', 'c_rostrum_protraction', -1),
            ('MN9', 'm9', 'c_rostrum_protraction', 1),
            ('MN3L', 'm3L', 'c_haustellum_extension', -1),
            ('MN3M', 'm3M', 'c_haustellum_extension', -1),
            ('MN4a', 'm4a', 'c_haustellum_extension', 1),
            ('MN4b', 'm4b', 'c_haustellum_extension', 1),
            ('MN6', 'm6', f'{s}_labellum_extension', 1),
            ('MN7', 'm7', f'{s}_labellum_abduction', 1),
        ]:
            add(neuron, 'pm', side, target, [(joint, -.025 * torque)], 30, MOUTH_SOURCE,
                'Published primary positioning action; MN2D/MN4 subdivisions retain MaleCNS names. Effective moment arm and force are uncalibrated.')
        for neuron in ['MN5', 'MN8', 'MN10', 'MN11D', 'MN11V', 'MN12D', 'MN13']:
            target = neuron[2:].lower()
            add(neuron, 'pm', side, 'pharyngeal_'+target,
                [('c_pharyngeal_'+target+'_dilation', -1.)], 2, MOUTH_SOURCE,
                'Elastic wall/valve displacement follows the anatomical insertion hypothesis. No fluid transport or ingestion is modeled. Bilateral neurons converge on a shared wall element.')
        add('CEM', 'pm', side, 'crop_duct', [('c_crop_duct_constriction', -1.)], 2,
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC11398398/',
            'Crop-innervating enteric motor output drives an elastic duct-wall proxy; no fluid transport, peristaltic sequence or digestion is modeled.')
        # TH1/TH2 are named transverse neck muscles. Yaw-only projection is an
        # explicit simplification of the cervical linkage (not measured torque).
        for neuron, target in [('ADNM1 MN', 'th1'), ('ADNM2 MN', 'th2')]:
            add(neuron, 'nm', side, target, [('c_head_turn', -.025 * sign)], 20, NECK_SOURCE,
                'Annotated TH1/TH2 identity; same-side yaw projection of transverse cervical muscle is uncalibrated.')
        for neuron, target in [('TTMn', 'ttm'), ('STTMm', 'satellite_ttm')]:
            add(neuron, 'wm', side, target, [(f'{s}m_trochanterfemur_pitch', -.05)], 1400, MOTOR_SOURCE,
                'Tergotrochanteral output extends the middle-leg coxa/trochanter joint. Satellite identity is tentative in the cross-specimen reference.')
        wing = f'{s}_wing'
        for neuron, target, terms in [
            ('DLMn a, b', 'dlm_ab', [('elevation', 1.)]),
            ('DLMn c-f', 'dlm_cf', [('elevation', 1.)]),
            ('DVMn 1a-c', 'dvm1', [('elevation', -1.)]),
            ('DVMn 2a, b', 'dvm2', [('elevation', -1.)]),
            ('DVMn 3a, b', 'dvm3', [('elevation', -1.)]),
            ('b1 MN', 'b1', [('spread', -1.)]),
            ('b2 MN', 'b2', [('spread', -1.)]),
            ('b3 MN', 'b3', [('spread', 1.)]),
            ('i1 MN', 'i1', [('spread', 1.)]),
            ('i2 MN', 'i2', [('spread', 1.)]),
            ('iii1 MN', 'iii1', [('spread', 1.), ('feather', -.3)]),
            ('iii3 MN', 'iii3', [('spread', 1.), ('feather', -.3)]),
            ('hg1 MN', 'hg1', [('feather', -1.)]),
            ('hg2 MN', 'hg2', [('feather', -1.)]),
            ('hg3 MN', 'hg3', [('feather', -1.)]),
            ('hg4 MN', 'hg4', [('feather', -1.)]),
            ('tp1 MN', 'tp1', [('elevation', .5), ('spread', .5)]),
            ('tp2 MN', 'tp2', [('elevation', .5), ('spread', .5)]),
            ('tpn MN', 'tpn', [('elevation', .5), ('spread', .5)]),
            ('ps1 MN', 'ps1', [('elevation', -.5), ('spread', .5)]),
            ('ps2 MN', 'ps2', [('elevation', -.5), ('spread', .5)]),
        ]:
            add(neuron, 'wm', side, target, [(wing+'_'+j, .025 * c) for j, c in terms],
                100 if target.startswith(('dlm', 'dvm')) else 20, MOTOR_SOURCE,
                'Named wing muscle; effective sclerite/wing-axis projection is an uncalibrated hypothesis. Hill force does not reproduce asynchronous power generation, phase-dependent steering, or aerodynamic flight.')
        for neuron, target, axis in [('hDVM MN', 'hdvm', 'stroke'), ('hi1 MN', 'hi1', 'trim'),
                                      ('hi2 MN', 'hi2', 'trim'), ('hiii2 MN', 'hiii2', 'trim')]:
            add(neuron, 'hm', side, target, [(f'{s}_haltere_{axis}', -.015)], 10, MOTOR_SOURCE,
                'Named haltere muscle; stroke/trim projection is an uncalibrated hypothesis. hi2/hiii2 matches are tentative. Asynchronous oscillation and gyroscopic sensory tuning are absent.')
    return tuple(specs)


MUSCLES = muscle_specs()
LOOKUP = {(m.neuron_type, m.subclass, m.side): 90 + i for i, m in enumerate(MUSCLES)}
CHANNEL_COUNT = 90 + len(MUSCLES)


def add_mechanics(root, bodies, parameters):
    def joint(body, name, axis, limits, stiffness=.5, damping=.005, kind='hinge'):
        ET.SubElement(body, 'joint', name=name, type=kind, axis=axis, range=limits,
                      stiffness=str(stiffness), damping=str(damping), armature='1e-6', springref='0')
    joint(bodies['c_head'], 'c_head_turn', '0 0 1', '-.25 .25', 2, .01)
    joint(bodies['c_rostrum'], 'c_rostrum_protraction', '0 -1 0', '0 1.1')
    joint(bodies['c_haustellum'], 'c_haustellum_extension', '0 1 0', '0 .9')
    # The source has a single haustellum mesh. These added effective labellar
    # pads sit at its distal tip; detailed furca/apodeme geometry is unavailable.
    for side, sign in [('l', 1), ('r', -1)]:
        body = ET.SubElement(bodies['c_haustellum'], 'body', name=side+'_labellum', pos=f'.315 {sign*.065} -.14')
        joint(body, side+'_labellum_extension', '0 1 0', '0 .7', .05, .0005)
        joint(body, side+'_labellum_abduction', f'0 0 {sign}', '0 .7', .05, .0005)
        ET.SubElement(body, 'geom', name=side+'_labellum_pad', type='capsule', fromto='0 0 0 .055 0 0',
                      size='.025', mass=str(1e-6*parameters.mass_scale), contype='2', conaffinity='3', group='1')
        joint(bodies[side+'_wing'], side+'_wing_feather', '0 1 0', '-.35 .35', .5, .005)
        joint(bodies[side+'_haltere'], side+'_haltere_stroke', '1 0 0', '-.6 .6', .15, .001)
        joint(bodies[side+'_haltere'], side+'_haltere_trim', '0 0 1', '-.2 .2', .15, .001)
    for i, name in enumerate(['5', '8', '10', '11d', '11v', '12d', '13']):
        body = ET.SubElement(bodies['c_rostrum'], 'body', name='c_pharyngeal_'+name,
                             pos=f'{-.03*i} 0 -.035')
        joint(body, 'c_pharyngeal_'+name+'_dilation', '0 0 -1', '0 .025', 20, .01, 'slide')
        ET.SubElement(body, 'geom', name='c_pharyngeal_'+name+'_wall', type='sphere', size='.01',
                      mass=str(1e-6*parameters.mass_scale), contype='0', conaffinity='0', group='4')
    crop = ET.SubElement(bodies['c_abdomen12'], 'body', name='c_crop_duct', pos='0 0 0')
    joint(crop, 'c_crop_duct_constriction', '0 0 1', '0 .025', 20, .01, 'slide')
    ET.SubElement(crop, 'geom', name='c_crop_duct_wall', type='sphere', size='.01',
                  mass=str(1e-6*parameters.mass_scale), contype='0', conaffinity='0', group='4')
    # Retracted mouthparts are nested inside the head. Preserve that intentional
    # envelope overlap; all external leg/wing and mouth/environment contacts remain.
    contact = root.find('contact')
    ET.SubElement(contact, 'exclude', body1='c_head', body2='c_haustellum')
    for side in ['l', 'r']:
        for other in ['c_head', 'c_rostrum']:
            ET.SubElement(contact, 'exclude', body1=other, body2=side+'_labellum')
    tendons, actuators = root.find('tendon'), root.find('actuator')
    for muscle in MUSCLES:
        tendon = ET.SubElement(tendons, 'fixed', name=muscle.name+'_transmission')
        for name, coefficient in muscle.joints:
            ET.SubElement(tendon, 'joint', joint=name, coef=str(coefficient))
        ET.SubElement(actuators, 'muscle', name=muscle.name, tendon=muscle.name+'_transmission',
                      lengthrange='-.15 .15', force=str(muscle.force*parameters.strength_scale),
                      timeconst='.002 .004', fpmax='.00001')
