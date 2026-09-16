"""Inspectable evidence and qualitative confidence, separate from routing.

Confidence concerns the target assignment and the simulated force path
separately. Neither is a probability or a validation of locomotion.
"""
import json
from pathlib import Path

TABLE = json.loads((Path(__file__).with_name('assets') / 'motor_target_evidence.json').read_text())
ANNOTATIONS = 'https://male-cns.janelia.org/download/'
NECK = 'https://www.nature.com/articles/s41586-024-07222-5'
LEG = 'https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf'


def annotate_evidence(record):
    name = record.get('type')
    matches = TABLE['types'].get(record.get('mancType') or name, [])
    # MaleCNS retains an MN suffix on these two published MANC names.
    if not matches and name in {'ADNM1 MN', 'ADNM2 MN'}:
        matches = TABLE['types'].get(name[:-3], [])
    sources = [{'title': 'MaleCNS v1.0 annotations', 'url': ANNOTATIONS,
                'locator': f"bodyId {record['body_id']}",
                'supports': 'Neuron identity, annotated class, side and cross-dataset type match.'}]
    if matches:
        sources.append({'title': 'Cheong et al. — motor matching table',
                        'url': TABLE['article'], 'locator': 'Supplementary file 3',
                        'supports': 'Cross-specimen muscle-target evidence; original target, certainty and caveats are retained.',
                        'data_url': TABLE['source']})
    function_source = record.get('function_source')
    if function_source and function_source not in {s['url'] for s in sources}:
        sources.append({'title': 'Gorko et al. — neck muscle anatomy' if function_source == NECK
                        else 'Azevedo et al. — leg muscle atlas' if function_source == LEG
                        else 'Muscle anatomy and function reference',
                        'url': function_source,
                        'locator': 'Figure 4h and Extended Data Figure 7' if function_source == NECK
                        else 'Table A1 and Figures A2–A17' if function_source == LEG else 'See mapping interpretation',
                        'supports': 'Muscle identity and/or anatomical interpretation; does not validate model force constants.'})
    mapped = record['status'] == 'mapped'
    level = 'supported' if mapped else 'unresolved'
    basis = 'Named annotation interpreted using the cited muscle reference; transfer across specimens and side assignment remain assumptions.'
    if name in {'MNwm35', 'FNM2', 'STTMm', 'hi2 MN', 'hiii2 MN'}:
        level = 'tentative'
        basis = 'The source explicitly gives a tentative or ambiguous cross-specimen muscle match.'
    elif name in {'MNhm42', 'MNhm43'}:
        level = 'family_only'
        basis = 'Source supports hb1 or hb2; it does not identify which muscle. The model preserves that ambiguity.'
    elif name in {'CvN4', 'CvN5', 'CvN6', 'CvN7'}:
        level = 'supported'
        basis = 'Gorko et al. Figure 4h directly identifies VL1 for these named neuron types; MaleCNS type correspondence is annotation-based.'
        if not any(s['url'] == NECK for s in sources):
            sources.append({'title': 'Gorko et al. — neck muscle anatomy', 'url': NECK,
                            'locator': 'Figure 4h', 'supports': 'CvN4–7 innervate VL1.'})
    elif record.get('reason') == 'muscle_action_unknown':
        level = 'supported'
        basis = 'The femur-reductor target is named, but its mechanical action is unresolved in the cited leg atlas.'
        sources.append({'title': 'Azevedo et al. — leg muscle atlas', 'url': LEG,
                        'locator': 'Table A1', 'supports': 'Identified muscle with unresolved action.'})
    elif not mapped:
        basis = 'No specific muscle target resolved by the implemented annotation and reference matching.'
    record.update(sources=sources, reference_matches=matches,
                  confidence={'identity': {'level': level, 'basis': basis},
                              'mechanics': {'level': 'approximate' if mapped else 'unresolved',
                                            'basis': record.get('routing') if mapped else 'No implemented force transmission.'}},
                  target_side=(('R' if record.get('somaSide') == 'L' else 'L')
                  if name == 'FNM2' else record.get('somaSide')) if mapped else None)
    return record
