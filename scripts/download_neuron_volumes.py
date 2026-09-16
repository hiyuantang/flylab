"""Extract MaleCNS volume annotations by ID with resumable column-range reads.

The public 4.6 GB Arrow object also contains unclassified segmentation fragments.
Only ID and size buffers are transferred; no brain neurons or edges are removed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
import numpy as np
import pyarrow as pa

OBJECT = 'v1.0/database/neuprint-inputs/Neuprint_Neurons.feather'
SOURCE = 'https://storage.googleapis.com/flyem-male-cns/' + OBJECT


def fetch(url, headers=None):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=45)
        except OSError:
            if attempt == 3: raise
            time.sleep(2**attempt)


class RemoteArrow(io.RawIOBase):
    def __init__(self, size, generation):
        self.size, self.pos = size, 0
        self.url = SOURCE + '?generation=' + generation
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        if n <= 0: return b''
        with fetch(self.url, {'Range': f'bytes={self.pos}-{self.pos+n-1}'}) as r:
            if r.status != 206 or r.headers['Content-Range'] != f'bytes {self.pos}-{self.pos+n-1}/{self.size}':
                raise ValueError('Remote range mismatch')
            value = r.read()
        if len(value) != n: raise ValueError('Incomplete range')
        self.pos += n
        return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, default=Path('data/full'))
    p.add_argument('--output', type=Path, default=Path('data/anatomy-reference/male-cns-volumes.npy'))
    args = p.parse_args()
    url = 'https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o/' + urllib.parse.quote(OBJECT, safe='')
    with fetch(url) as r: source = json.load(r)
    ids = np.load(args.graph/'body_ids.npy')
    identity = hashlib.sha256(ids.tobytes()).hexdigest()
    state_path = args.output.with_suffix('.progress.npz')
    volumes, done = np.full(len(ids), np.nan), set()
    if state_path.exists():
        prior = np.load(state_path)
        if str(prior['generation']) != source['generation'] or str(prior['ids_hash']) != identity:
            raise ValueError('Resume source or neuron IDs changed')
        volumes = prior['volumes']; done = set(prior['done'].tolist())
    local = threading.local()
    def reader():
        if not hasattr(local, 'reader'):
            file = RemoteArrow(int(source['size']), source['generation'])
            schema = pa.ipc.open_file(file).schema
            options = pa.ipc.IpcReadOptions(included_fields=[schema.get_field_index('bodyId:long'), schema.get_field_index('size:long')])
            local.reader = pa.ipc.open_file(file, options=options)
        return local.reader
    batches = reader().num_record_batches
    def batch(i):
        b = reader().get_batch(i)
        body, size = [b.column(k).to_numpy(zero_copy_only=False) for k in (0,1)]
        ix = np.searchsorted(ids, body).clip(0, len(ids)-1)
        valid = (ids[ix] == body) & np.isfinite(size) & (size > 0)
        return i, ix[valid], size[valid]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = state_path.with_suffix('.tmp.npz')
        np.savez(tmp, volumes=volumes, done=np.array(sorted(done)), generation=source['generation'], ids_hash=identity)
        tmp.replace(state_path)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(batch, i) for i in range(batches) if i not in done]
            for future in as_completed(futures):
                i, ix, values = future.result(); volumes[ix] = values; done.add(i)
                if len(done) % 50 == 0:
                    save(); print(f'{len(done)}/{batches} batches; {np.isfinite(volumes).sum()}/{len(ids)} measured volumes', flush=True)
    finally: save()
    np.save(args.output, volumes)
    args.output.with_suffix('.json').write_text(json.dumps({'source': SOURCE, 'generation': source['generation'],
        'source_md5': source.get('md5Hash'), 'field': 'size:long', 'units': 'segmentation voxels',
        'body_ids_sha256': identity, 'measured': int(np.isfinite(volumes).sum()), 'total': len(ids),
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2)+'\n')
    print('Complete', args.output, flush=True)


if __name__ == '__main__': main()
