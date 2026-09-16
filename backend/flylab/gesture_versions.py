"""Small JSON index beside self-contained adapter checkpoints; no base graph copies."""
import json
from pathlib import Path


def version_metadata(directory):
    result = []
    for path in Path(directory).glob('gesture-*.json'):
        try:
            item = json.loads(path.read_text())
            if item.get('id') == path.with_suffix('.pt').name and path.with_suffix('.pt').exists():
                result.append(item)
        except (ValueError, OSError):
            continue
    indexed = {x['id'] for x in result}
    # Older experiments remain visible and deletable, but cannot initialize the
    # new muscle-gradient trainer. They are never silently treated as compatible.
    import torch
    from datetime import datetime, timezone
    for path in Path(directory).glob('gesture-*.pt'):
        if path.name in indexed:
            continue
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('version') in {'gesture-muscle-adam-v2', 'gesture-muscle-adam-v3', 'gesture-muscle-adam-v4', 'gesture-muscle-adam-v5'} and 'metadata' in payload:
                result.append({**payload['metadata'], 'bytes': path.stat().st_size})
                continue
            results = payload.get('results', {})
            history = results.get('history', [])
            result.append({'id': path.name, 'chart_id': 'legacy-' + path.stem,
                'chart_name': 'Earlier pose search', 'parent': None, 'version': 1,
                'name': 'Legacy SPSA adapter · pose targets', 'compatible': False,
                'created_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                'rank': payload.get('rank', 2), 'iterations': results.get('total', 0),
                'batch_size': 3, 'horizon': results.get('horizon', 25), 'learning_rate': 0,
                'seed': payload.get('seed', 42), 'final_loss': history[-1]['loss'] if history else 0,
                'wall_seconds': results.get('wall_seconds', 0), 'bytes': path.stat().st_size,
                'trainable_parameters': payload['parameters'].numel(),
                'proportions': {'palm': 33.3, 'fist': 33.3, 'point': 33.3},
                'optimizer_start': 'Earlier SPSA experiment', 'behavior_validated': results.get('success', False)})
        except (ValueError, OSError, RuntimeError, KeyError, AttributeError):
            continue
    from .gesture_training import compatible_inference
    for item in result:
        if item.get('compatible') is False:
            continue
        payload = item
        if item.get('implementation_sha256') is None:
            payload = torch.load(Path(directory) / item['id'], map_location='cpu', weights_only=True)
        item['compatible'] = compatible_inference(payload)
    return sorted(result, key=lambda x: x['created_at'])


def descendants(versions, version):
    lookup = {x['id']: x for x in versions}
    if version not in lookup:
        raise ValueError('Weight version no longer exists')
    result, pending = [], [version]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.append(current)
        pending.extend(x['id'] for x in versions if x.get('parent') == current)
    return sorted(result)


def stratified_gestures(rng, proportions, batch_size):
    """Randomized systematic strata: expected counts match requested proportions.

    Every batch count differs from its ideal count by at most one. Random order
    prevents the preview from always displaying the same gesture class.
    """
    import numpy as np
    from .gesture_scene import GESTURES
    probabilities = np.array([proportions.get(g, 0.) for g in GESTURES], dtype=float)
    probabilities /= probabilities.sum()
    points = (rng.random() + np.arange(batch_size)) / batch_size
    indices = np.searchsorted(np.cumsum(probabilities), points, side='right')
    return rng.permutation(np.array(list(GESTURES))[indices])
