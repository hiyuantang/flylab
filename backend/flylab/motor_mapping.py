"""Versioned motor target interpretations; anatomical labels are never guessed.

Functional actions: Azevedo et al. 2024 supplement, Table A1/Figures A2-A17.
Model-axis projections below are explicit engineering approximations, not muscle
insertion reconstructions. Paired legs use mirrored axes in body.make_xml.
"""
from collections import Counter
from .peripheral_mechanics import PROFILE as PERIPHERAL_PROFILE, LOOKUP, MUSCLES

LEGACY_PROFILE = 'leg-routing-v1'
MAPPING_PROFILE = 'muscle-routing-v2'
PRETARSAL_PROFILE = 'muscle-routing-v3'
PROFILES = (LEGACY_PROFILE, MAPPING_PROFILE, PRETARSAL_PROFILE, PERIPHERAL_PROFILE)
LONG_TENDON_TARGETS = {'ltm MN', 'ltm1-tibia MN', 'ltm2-femur MN'}
SOURCE = 'https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf'

# (link, axis, polarity, contribution). Polarity 0 is positive joint torque.
LEGACY_TARGETS = {
    'Ti flexor MN': ('tibia', 'pitch', 1),
    'Acc. ti flexor MN': ('tibia', 'pitch', 1),
    'Ti extensor MN': ('tibia', 'pitch', 0),
    'Tr flexor MN': ('trochanterfemur', 'pitch', 1),
    'Acc. tr flexor MN': ('trochanterfemur', 'pitch', 1),
    'Tr extensor MN': ('trochanterfemur', 'pitch', 0),
    'Ta depressor MN': ('tarsus1', 'pitch', 1),
    'Ta levator MN': ('tarsus1', 'pitch', 0),
}
TARGETS = {name: ((link, axis, sign, 1.),) for name, (link, axis, sign) in LEGACY_TARGETS.items()}
TARGETS.update({
    'Ti flexor MN': (('tibia', 'pitch', 0, 1.),),
    'Acc. ti flexor MN': (('tibia', 'pitch', 0, 1.),),
    'Ti extensor MN': (('tibia', 'pitch', 1, 1.),),
    'Ta depressor MN': (('tarsus1', 'pitch', 0, 1.),),
    'Ta levator MN': (('tarsus1', 'pitch', 1, 1.),),
    'Tergopleural/Pleural promotor MN': (('coxa', 'pitch', 1, 1.),),
    'Sternal anterior rotator MN': (('coxa', 'pitch', 1, 1.),),
    'Sternal posterior rotator MN': (('coxa', 'pitch', 0, 1.),),
    'Sternal adductor MN': (('coxa', 'yaw', 1, 1.),),
    # Named action has two components. Equal splitting is an uncalibrated
    # functional projection; it does not claim equal biological moment arms.
    'Pleural remotor/abductor MN': (('coxa', 'pitch', 0, .5), ('coxa', 'yaw', 0, .5)),
    'Sternotrochanter MN': (('trochanterfemur', 'pitch', 0, 1.),),
    'Tergotr. MN': (('trochanterfemur', 'pitch', 0, 1.),),
})
BODY_REGIONS = {'fl': 'front leg', 'ml': 'middle leg', 'hl': 'hind leg',
                'wm': 'wing/thoracic flight', 'hm': 'haltere', 'ad': 'abdomen',
                'nm': 'neck', 'pm': 'mouthparts', 'am': 'antenna', 'rm': 'head rm class', 'xm': 'thoracic xm class'}
# Only these exit nerves unambiguously identify a leg neuromere. Others,
# including AbN1 supplying some hind-leg targets, are preserved as evidence.
NERVE_LEG = {'ProLN': 'fl', 'DProN': 'fl', 'VProN': 'fl', 'ProAN': 'fl',
             'MesoLN': 'ml', 'MetaLN': 'hl'}


def resolve_motor(row, profile=MAPPING_PROFILE):
    if profile not in PROFILES:
        raise ValueError('Unknown muscle mapping profile')
    record = {key: row.get(key) for key in ['type', 'superclass', 'subclass', 'somaSide', 'exitNerve', 'instance', 'mancType']}
    record.update(body_id=int(row['bodyId']), body_region=BODY_REGIONS.get(row.get('subclass'), 'unresolved'),
                  target_label=row.get('type'), source='MaleCNS v1.0 neuron annotations')
    name, subclass, side = row.get('type'), row.get('subclass'), row.get('somaSide')
    if profile == PERIPHERAL_PROFILE and row.get('superclass') in {'cb_motor', 'vnc_motor'} and (name, subclass, side) in LOOKUP:
        channel = LOOKUP[(name, subclass, side)]
        muscle = MUSCLES[channel - 90]
        instance = str(row.get('instance') or '')
        if instance.endswith(('_L', '_R')) and instance[-1] != side:
            record.update(status='unmapped', reason='annotation_conflict', projections=[])
        else:
            record.update(status='mapped', reason=None, leg=side, projections=[],
                          peripheral_channel=channel, muscle_target=muscle.target,
                          function_source=muscle.source, evidence='Named MaleCNS muscle target; same-side assignment is assumed.',
                          routing=muscle.interpretation)
        return record
    reason = None
    target = TARGETS.get(name)
    if profile == LEGACY_PROFILE:
        old = LEGACY_TARGETS.get(name)
        target = ((*old, 1.),) if old else None
    if profile in {PRETARSAL_PROFILE, PERIPHERAL_PROFILE} and name in LONG_TENDON_TARGETS:
        target = (('pretarsus', 'flexion', 0, 1.),)
    if row.get('superclass') != 'vnc_motor' or subclass not in {'fl', 'ml', 'hl'}:
        reason = 'peripheral_route_unresolved' if profile == PERIPHERAL_PROFILE else 'body_actuator_missing'
    elif not name or (name not in TARGETS and not (profile in {PRETARSAL_PROFILE, PERIPHERAL_PROFILE} and name in LONG_TENDON_TARGETS)):
        reason = ('muscle_action_unknown' if name == 'Fe reductor MN' else
                  'pretarsal_tendon_missing' if str(name).startswith('ltm') else 'muscle_identity_unresolved')
    elif target is None:
        reason = 'legacy_profile_excluded'
    elif side not in {'L', 'R'}:
        reason = 'side_unresolved'
    elif profile != LEGACY_PROFILE:
        instance = str(row.get('instance') or '')
        instance_side = instance[-1] if instance.endswith(('_L', '_R')) else None
        nerve_leg = NERVE_LEG.get(row.get('exitNerve'))
        if (instance_side and instance_side != side) or (nerve_leg and nerve_leg != subclass):
            reason = 'annotation_conflict'
    if reason:
        record.update(status='unmapped', reason=reason, projections=[])
        return record
    record.update(status='mapped', reason=None, leg=side+{'fl': 'F', 'ml': 'M', 'hl': 'H'}[subclass],
                  projections=target, function_source=SOURCE,
                  evidence='Named muscle target and leg subclass; somaSide implies ipsilateral muscle. Instance/exit nerve checked where informative.',
                  routing='Fixed-axis functional projection; strength, rate scaling, co-contraction, insertion geometry and side assignment remain assumptions.')
    if name in LONG_TENDON_TARGETS:
        record['routing'] = 'Distal spatial tendon pulling a paired-claw hinge; proximal tendon path, geometry, strength and elastic return are uncalibrated approximations. No adhesive pad force.'
    return record


def inventory_summary(records):
    regions = {}
    for record in records:
        region = regions.setdefault(record['body_region'], {'total': 0, 'mapped': 0, 'unmapped': 0})
        region['total'] += 1
        region[record['status']] += 1
    return {'total_motor_neurons': len(records), 'coverage_by_region': regions,
            'unmapped_reasons': dict(Counter(r['reason'] for r in records if r['status'] == 'unmapped'))}


def validate_body_profile(profile, parameters):
    if profile not in PROFILES:
        raise ValueError('Unknown muscle mapping profile')
    expected = 'peripheral-v1' if profile == PERIPHERAL_PROFILE else 'pretarsal-v1' if profile == PRETARSAL_PROFILE else 'baseline'
    if profile not in PROFILES or parameters.appendage_model != expected:
        raise ValueError('Muscle mapping and physical appendage model do not match')
