"""Replay GPU destination marking with cooperative thread counts, offline only."""
import json
from pathlib import Path
import statistics
import time
import torch
from flylab.full_connectome import FullBrain
from flylab.metal_dynamics import ActiveRowMetalDynamics, kernels


def main():
    torch.set_num_threads(1)
    brain = FullBrain(Path('data/full'))
    engine = ActiveRowMetalDynamics(brain.weights)
    libraries = {lanes: kernels(torch.float16, torch.float16, active_rows=True, mark_lanes=lanes)
                 for lanes in [1, 4, 8, 32]}
    results = {}
    for stride in [2000, 20, 1]:
        emitted = engine.zero.clone()
        emitted[::stride] = 1
        reference = None
        rows = {}
        for lanes, lib in libraries.items():
            def call():
                engine.active.zero_()
                lib.mark_active(engine.out_ptr, engine.out_post, emitted, engine.active, threads=engine.n*lanes)
            call()
            torch.mps.synchronize()
            flags = engine.active.cpu().clone()
            if reference is None:
                reference = flags
            assert torch.equal(reference, flags)
            timings = []
            for _ in range(3):
                torch.mps.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    call()
                torch.mps.synchronize()
                timings.append((time.perf_counter()-start)*1000/20)
            rows[lanes] = {'median_clear_and_mark_ms': statistics.median(timings), 'samples_ms': timings}
            print(stride, lanes, rows[lanes]['median_clear_and_mark_ms'], flush=True)
        results[stride] = {'emitting_sources': len(emitted[::stride]), 'marking': rows}
    Path('docs/results/fp16-marking-profile.json').write_text(json.dumps({
        'neurons':engine.n, 'edges':engine.pre.numel(), 'results':results,
        'limits':'Fixed synthetic emitted buffers; synchronized wall time includes clearing flags and host dispatch. Not a full simulation benchmark.'},indent=2)+'\n')


if __name__ == '__main__':
    main()
