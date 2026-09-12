"""Read-only spatial/adjacency views of the complete simulated graph.

Coordinates are source somaLocation values, never inferred from cell class.
Only rendering payloads are float32; simulation tensors remain unchanged.
"""
from collections import OrderedDict
import numpy as np


class NeuralView:
    def __init__(self, brain):
        self.brain = brain
        self.ids = brain.ids
        self.positions = np.full((len(self.ids), 3), np.nan, dtype='<f4')
        self.vnc = np.zeros(len(self.ids), dtype=np.uint8)
        for i, row in enumerate(brain.neurons):
            value = row.get('somaLocation')
            if value is not None and len(value) == 3 and np.isfinite(value).all():
                self.positions[i] = value
            superclass = str(row.get('superclass') or '')
            self.vnc[i] = superclass.startswith('vnc_') or superclass in {'ascending_neuron', 'efferent_ascending'}
        self.valid = np.isfinite(self.positions).all(axis=1)
        # Column-major payload: float64 IDs, float32 XYZ, uint8 VNC-class flag.
        self.layout = np.asarray(self.ids, dtype='<f8').tobytes() + self.positions.tobytes() + self.vnc.tobytes()
        self.ptr = np.load(brain.directory / 'indptr.npy', mmap_mode='r')
        self.pre = np.load(brain.directory / 'indices.npy', mmap_mode='r')
        self.counts = np.load(brain.directory / 'counts.npy', mmap_mode='r')
        self.cache = OrderedDict()

    def metadata(self):
        return {'neurons': len(self.ids), 'positioned': int(self.valid.sum()),
                'missing_positions': int((~self.valid).sum()), 'graph_sha256': self.brain.manifest['graph_sha256'],
                'coordinates': 'MaleCNS EM voxel coordinates, 8 nm per unit; somaLocation only',
                'source': 'https://male-cns.janelia.org/download/',
                'limits': 'Soma points are not neuron arbors. Lines represent measured directed neuron pairs, not axon trajectories or synapse locations.'}

    def adjacency(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        start, end = self.ptr[i:i+2]
        incoming = np.arange(start, end, dtype=np.int64)
        # One vectorized scan on first selection; bounded LRU avoids storing a
        # second 25M-edge graph. Subsequent polling never rescans connectivity.
        outgoing = np.flatnonzero(self.pre == i)
        result = []
        for offsets, direction in [(incoming, 'incoming'), (outgoing, 'outgoing')]:
            others = self.pre[offsets] if direction == 'incoming' else np.searchsorted(self.ptr, offsets, side='right')-1
            order = np.lexsort((self.ids[others], -self.counts[offsets]))
            result.append((others[order], self.counts[offsets][order]))
        self.cache[i] = result
        if len(self.cache) > 32:
            self.cache.popitem(last=False)
        return result

    def neuron(self, body_id, direction='incoming', offset=0, limit=100):
        i = self.brain.lookup.get(body_id)
        if i is None:
            raise ValueError('Neuron ID is not in the simulated graph')
        incoming, outgoing = self.adjacency(i)
        others, counts = incoming if direction == 'incoming' else outgoing
        rates = self.brain.rates.detach().cpu().numpy()
        def item(j):
            row = self.brain.neurons[j]
            return {'body_id': int(self.ids[j]), 'index': int(j), 'type': row.get('type'),
                    'superclass': row.get('superclass'), 'transmitter': row.get('transmitter'),
                    'soma_side': row.get('somaSide'), 'position': self.positions[j].tolist() if self.valid[j] else None,
                    'rate_hz': float(rates[j])}
        selected = item(i)
        selected.update(voltage=float(self.brain.voltage[i]),
                        voltage_unit='mV' if self.brain.config.profile == 'shiu-2024' else 'normalized',
                        spike_count=int(self.brain.spike_counts[i]))
        return {'neuron': selected, 'direction': direction, 'offset': offset, 'limit': limit,
                'total': len(others), 'incoming_total': len(incoming[0]), 'outgoing_total': len(outgoing[0]),
                'incoming_synapses': int(incoming[1].sum()), 'outgoing_synapses': int(outgoing[1].sum()),
                'partners': [{**item(int(j)), 'synapses': int(w)} for j, w in zip(others[offset:offset+limit], counts[offset:offset+limit])],
                'neural_time_ms': self.brain.time_ms, 'graph_sha256': self.brain.manifest['graph_sha256']}
