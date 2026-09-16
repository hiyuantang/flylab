"""Isolated FP16 kernel replay timings; no live state or checkpoint access.

Repeated dispatches include CPU submission overhead; they are not hardware
counter measurements or a representative full-simulation timing breakdown.
"""
import json
from pathlib import Path
import time
import torch
from flylab.full_connectome import FullBrain
from flylab.metal_dynamics import MetalDynamics


def main():
    torch.set_num_threads(1)
    brain = FullBrain(Path('data/full'), device='mps')
    brain.metal = MetalDynamics(brain.weights, weight_dtype=torch.float16, state_dtype=torch.float16)
    brain.dtype = torch.float16
    brain.reset()
    n = len(brain.ids)
    drive = torch.full((n,), 14.7, device='mps', dtype=torch.float16)
    mask = torch.zeros(n, device='mps', dtype=torch.bool)
    params = torch.tensor([.995, .98, .005, .005, .998, 19.98], device='mps', dtype=torch.float16)
    engine = brain.metal
    sparse = engine.zero.clone()
    sparse[::2000] = 1  # 84 deterministic sources, independent of their type.
    burst = engine.zero.clone()
    burst[::20] = 1
    calls = {
        'integrate': lambda: engine.lib.integrate(brain.voltage, brain.current, brain.spikes,
            brain.last_spike_tick, brain.refractory_ticks, drive, mask, brain.output_gain,
            brain.delay_queue[0], engine.free, 100, params, threads=n),
        'gather_zero': lambda: engine.deliver(engine.zero),
        'gather_84_sources': lambda: engine.deliver(sparse),
        'gather_8335_sources': lambda: engine.deliver(burst),
        'finish': lambda: engine.lib.finish(brain.voltage, brain.current, brain.spikes,
            engine.free, engine.incoming, engine.zero, brain.delay_queue[0],
            brain.spike_counts, brain.rates, params, threads=n),
    }
    rows = {}
    for name, call in calls.items():
        for _ in range(3):
            call()
        torch.mps.synchronize()
        wall = time.perf_counter()
        for _ in range(50):
            call()
        torch.mps.synchronize()
        rows[name] = {'mean_wall_ms': (time.perf_counter()-wall)*1000/50}
        print(name, rows[name], flush=True)
    result = {'neurons': n, 'edges': brain.weights._nnz(), 'repeats': 50,
              'kernel': engine.version, 'phases': rows,
              'limits': 'Synthetic repeated kernel dispatches on fixed buffers, not a summed simulation profile. Wall time includes submission gaps. Repeated integration/finish changes scratch state. No Instruments hardware counters collected.'}
    Path('docs/results/fp16-profile.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
