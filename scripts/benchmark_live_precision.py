"""Compare complete workbench cycles from copies of one saved state.

Includes senses, neural updates, muscle/body integration and snapshot generation;
excludes checkpoint loading, kernel compilation, HTTP transport and browser work.
Never writes the input checkpoint or changes the running animal.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from flylab.live_state import load_live, NEURAL_TENSORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    parser.add_argument('--cycles', type=int, default=60)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('docs/results/live-fp16.json'))
    args = parser.parse_args()
    if args.cycles < 1 or args.trials < 1:
        parser.error('cycles and trials must be positive')
    results = []
    for trial in range(args.trials):
        for precision in (['float32', 'float16'] if trial % 2 == 0 else ['float16', 'float32']):
            sim = load_live(args.checkpoint, args.graph)
            sim.full_brain.set_device('mps', precision)
            # Prime lazy sensor routing and shader execution outside timing.
            sim.advance()
            sim.snapshot()
            start_sim = sim.time
            start_spikes = sim.full_brain.total_spikes
            cycles, neural = [], []
            for _ in range(args.cycles):
                torch.mps.synchronize()
                start = time.perf_counter()
                sim.advance()
                snapshot = sim.snapshot()
                torch.mps.synchronize()
                cycles.append(time.perf_counter() - start)
                neural.append(sim.full_brain.last_wall_seconds)
            if any(not torch.isfinite(getattr(sim.full_brain, k)).all() for k in NEURAL_TENSORS):
                raise RuntimeError('Non-finite neural endpoint')
            elapsed, simulated = sum(cycles), sim.time - start_sim
            result = dict(trial=trial, precision=precision, execution=sim.full_brain.execution_summary(),
                          graph_sha256=sim.full_brain.manifest['graph_sha256'],
                          neurons=len(sim.full_brain.ids), edges=sim.full_brain.manifest['edges'],
                          scene=sim.body.scene_id, command_hz=sim.command_hz,
                          start_simulated_seconds=start_sim, simulated_seconds=simulated,
                          wall_seconds=elapsed, wall_seconds_per_simulated_second=elapsed/simulated,
                          cycles_per_wall_second=args.cycles/elapsed,
                          neural_wall_seconds=sum(neural), mean_cycle_ms=1000*statistics.mean(cycles),
                          total_new_spikes=sim.full_brain.total_spikes-start_spikes,
                          cycle_wall_seconds=cycles)
            results.append(result)
            print(json.dumps({k: result[k] for k in ['trial', 'precision', 'wall_seconds', 'simulated_seconds', 'cycles_per_wall_second', 'neural_wall_seconds']}), flush=True)
            del snapshot, sim
            gc.collect()
            torch.mps.empty_cache()
    report = dict(scope='Complete simulation cycles including snapshots; excludes transport/browser, loading and one warm-up cycle.',
                  torch_version=torch.__version__, trials=results)
    medians = {mode: {key: statistics.median(r[key] for r in results if r['precision'] == mode)
                     for key in ['wall_seconds_per_simulated_second', 'mean_cycle_ms',
                                 'cycles_per_wall_second', 'neural_wall_seconds']}
               for mode in ['float32', 'float16']}
    baseline = medians['float32']['wall_seconds_per_simulated_second']
    candidate = medians['float16']['wall_seconds_per_simulated_second']
    report['summary'] = {'medians': medians, 'speedup': baseline / candidate,
                         'wall_time_reduction_percent': 100 * (1 - candidate / baseline)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
