"""Read-only anatomical/wiring audit; writes only the requested aggregate report.

Run: PYTHONPATH=backend uv run python scripts/audit_malecns.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lookup(sorted_ids, values):
    """Indices in sorted unique IDs, with -1 for unknown segment IDs."""
    if not len(sorted_ids):
        return np.full(len(values), -1, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, values)
    safe = np.minimum(pos, len(sorted_ids) - 1)
    return np.where((pos < len(sorted_ids)) & (sorted_ids[safe] == values), pos, -1)


def tally(codes, weights, size):
    """Count rows and sum weights in int64, without floating-point reduction."""
    rows = np.bincount(codes, minlength=size)
    synapses = np.zeros(size, dtype=np.int64)
    np.add.at(synapses, codes, weights)
    return np.stack((rows, synapses), axis=1)


def layer_codes(pre, post):
    """Forward, same rank, backward, or missing rank; not anatomical layers."""
    return np.where(~np.isfinite(pre) | ~np.isfinite(post), 3,
                    np.where(post > pre, 0, np.where(post == pre, 1, 2)))


def audit(directory):
    manifest = json.loads((directory / 'full/manifest.json').read_text())
    supplement = directory / 'anatomy-reference'
    sources = json.loads((supplement / 'sources.json').read_text())
    for item in manifest['sources']:
        if digest(directory / item['file']) != item['sha256']:
            raise ValueError(f"Source hash mismatch: {item['file']}")
    for item in sources['files']:
        if digest(supplement / item['name']) != item['sha256']:
            raise ValueError(f"Supplement hash mismatch: {item['name']}")
    arrays = ['body_ids.npy', 'indptr.npy', 'indices.npy', 'counts.npy']
    graph_hash = hashlib.sha256(''.join(digest(directory / 'full' / p) for p in arrays).encode()).hexdigest()
    if graph_hash != manifest['graph_sha256']:
        raise ValueError('Imported graph hash mismatch')
    ann = sorted(feather.read_table(directory / 'body-annotations.feather').to_pylist(), key=lambda r: r['bodyId'])
    ann_ids = np.array([r['bodyId'] for r in ann], dtype=np.int64)
    if len(np.unique(ann_ids)) != len(ann_ids):
        raise ValueError('Duplicate annotation IDs')
    included = [r for r in ann if r['superclass'] is not None]
    excluded = [r for r in ann if r['superclass'] is None]
    ids, ptr, indices, counts = [np.load(directory / 'full' / p, mmap_mode='r') for p in arrays]
    if not np.array_equal(ids, [r['bodyId'] for r in included]):
        raise ValueError('Imported neuron IDs differ from declared selection')
    n = len(ids)
    if len(ptr) != n + 1 or ptr[0] != 0 or ptr[-1] != len(indices) or np.any(np.diff(ptr) < 0):
        raise ValueError('Invalid CSR pointers')
    if len(indices) != len(counts) or np.any(indices < 0) or np.any(indices >= n) or np.any(counts <= 0):
        raise ValueError('Invalid CSR endpoints or counts')
    hex_rows = [r for r in included if r['assignedOlHex1'] is not None and r['assignedOlHex2'] is not None]
    hex_keys = Counter((r['somaSide'], r['type'], r['assignedOlHex1'], r['assignedOlHex2']) for r in hex_rows)
    status = Counter(r['statusLabel'] or '(missing)' for r in excluded)
    labels = ['included neuron', 'unannotated segment'] + ['excluded annotation: ' + s for s in sorted(status)]
    label_index = {v: i for i, v in enumerate(labels)}
    ann_category = np.array([0 if r['superclass'] is not None else label_index['excluded annotation: ' + (r['statusLabel'] or '(missing)')] for r in ann])
    raw = np.zeros((len(labels)**2, 2), dtype=np.int64)
    with pa.memory_map(str(directory / 'connectome-weights.feather'), 'r') as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            codes = []
            for col in ['body_pre', 'body_post']:
                pos = lookup(ann_ids, batch.column(col).to_numpy())
                codes.append(np.where(pos < 0, 1, ann_category[np.maximum(pos, 0)]))
            raw += tally(codes[0] * len(labels) + codes[1], batch.column('weight').to_numpy(), len(raw))
    print('Raw segment boundary audit complete', flush=True)
    classes = sorted({r['superclass'] for r in included})
    class_idx = {v: i for i, v in enumerate(classes)}
    category = np.array([class_idx[r['superclass']] for r in included])
    ranks = np.full(n, np.nan)
    traversal = feather.read_table(supplement / 'sensory_network_traversal_model_layers.feather')
    traversal_ids = traversal['node'].to_numpy()
    if len(np.unique(traversal_ids)) != len(traversal_ids):
        raise ValueError('Duplicate traversal IDs')
    positions = lookup(ids, traversal_ids)
    keep = positions >= 0
    ranks[positions[keep]] = traversal['layer_median'].to_numpy()[keep]
    blocks = np.zeros((len(classes)**2, 2), dtype=np.int64)
    layers = np.zeros((4, 2), dtype=np.int64)
    # Keys preserve CSR order, enabling exact reverse-edge lookup without a dense matrix.
    posts = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    keys = posts * n + indices
    if np.any(np.diff(keys) <= 0):
        raise ValueError('CSR is not sorted with unique directed pairs')
    reciprocal = self_edges = 0
    examples = []
    for start in range(0, len(indices), 500_000):
        end = min(start + 500_000, len(indices))
        pre, post, weight = indices[start:end], posts[start:end], counts[start:end]
        blocks += tally(category[pre] * len(classes) + category[post], weight, len(blocks))
        layers += tally(layer_codes(ranks[pre], ranks[post]), weight, 4)
        reverse = lookup(keys, pre.astype(np.int64) * n + post)
        mutual = (reverse >= 0) & (pre != post)
        reciprocal += int(mutual.sum())
        self_edges += int((pre == post).sum())
        if len(examples) < 5:
            for j in np.flatnonzero(mutual & (pre < post))[:5-len(examples)]:
                examples.append({'pre': int(ids[pre[j]]), 'post': int(ids[post[j]]), 'synapses': int(weight[j]), 'reverse_synapses': int(counts[reverse[j]])})
    with (supplement / 'dnan_cluster_function_20260509.csv').open() as stream:
        dn_rows = list(csv.DictReader(stream))
    included_set = set(map(int, ids))
    joined_dn = [r for r in dn_rows if int(r['bodyId']) in included_set]
    def matrix_rows(values, names):
        return [{'pre': names[i // len(names)], 'post': names[i % len(names)], 'connection_rows': int(v[0]), 'synapses': int(v[1])} for i, v in enumerate(values) if v[0]]
    raw_matrix = raw.reshape(len(labels), len(labels), 2)
    retained = raw_matrix[0, 0]
    boundary = raw_matrix[0, 1:].sum(axis=0) + raw_matrix[1:, 0].sum(axis=0)
    if retained.tolist() != [len(indices), int(counts.sum())]:
        raise ValueError('Source/import aggregate mismatch')
    if int(raw[:, 0].sum()) != manifest['source_connection_rows'] or int(raw[:, 1].sum()) != manifest['source_synapses']:
        raise ValueError('Source totals differ from import manifest')
    return {
        'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset': manifest['dataset'], 'source_hashes_verified': True,
        'graph_sha256': graph_hash, 'sources': manifest['sources'], 'supplement': sources,
        'selection': manifest['selection'],
        'annotation_rows': len(ann), 'included_neurons': n,
        'excluded_annotation_rows': len(excluded), 'excluded_status_counts': dict(status),
        'unique_type_labels': len({r['type'] for r in included if r['type'] is not None}),
        'included_without_type': sum(r['type'] is None for r in included),
        'superclass_neurons': dict(Counter(r['superclass'] for r in included)),
        'hex_column_neurons': len(hex_rows),
        'hex_type_neurons': dict(Counter(r['type'] for r in hex_rows)),
        'hex_duplicate_side_type_coordinate_groups': sum(v > 1 for v in hex_keys.values()),
        'raw_source_connection_rows': int(raw[:, 0].sum()), 'raw_source_synapses': int(raw[:, 1].sum()),
        'retained_connections': int(retained[0]), 'retained_synapses': int(retained[1]),
        'boundary_connection_rows': int(boundary[0]), 'boundary_synapses': int(boundary[1]),
        'segment_category_connections': matrix_rows(raw, labels),
        'superclass_connections': matrix_rows(blocks, classes),
        'self_connections': self_edges, 'directed_connections_with_reverse': reciprocal,
        'reciprocal_pair_examples': examples,
        'traversal_neurons_joined': int(np.isfinite(ranks).sum()),
        'traversal_source_rows': len(traversal),
        'traversal_direction': {name: {'connections': int(v[0]), 'synapses': int(v[1])} for name, v in zip(['forward', 'same_rank', 'backward', 'missing_rank'], layers)},
        'dnan_source_rows': len(dn_rows), 'dnan_joined_neurons': len({r['bodyId'] for r in joined_dn}),
        'dnan_with_function_annotation': len({r['bodyId'] for r in joined_dn if r['function'].strip()}),
        'limitations': ['Type labels can be composite; a label count is not a validated biological type count.', 'Superclass blocks are not neuropil boundaries.', 'Traversal rank is not an anatomical layer.', 'Source/import totals and hashes are checked; this audit is not a new edge-by-edge reimport comparison.', 'Structural preservation does not measure physiological or behavioral fidelity.'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/malecns-architecture-audit.json'))
    args = parser.parse_args()
    result = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['included_neurons', 'retained_connections', 'retained_synapses', 'directed_connections_with_reverse', 'traversal_direction']}, indent=2))
