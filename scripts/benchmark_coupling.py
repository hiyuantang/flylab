"""Measure complete FP16 cycles and verify identical delayed execution endpoints.

Loads independent copies; never overwrites the input checkpoint or live animal.
The delayed serial control changes only scheduling, not the pipeline's equations.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import numpy as np
import mujoco
import torch
import flylab.simulation as module
from flylab.coupling import run_pair
from flylab.live_state import load_live, NEURAL_TENSORS, STATE_SPEC


def endpoint(sim):
    digest = hashlib.sha256()
    for name in NEURAL_TENSORS:
        digest.update(getattr(sim.full_brain, name).detach().cpu().numpy().tobytes())
    state = np.empty(mujoco.mj_stateSize(sim.body.model, STATE_SPEC))
    mujoco.mj_getState(sim.body.model, sim.body.data, state, STATE_SPEC)
    digest.update(state.tobytes())
    digest.update(sim.pending_command.tobytes())
    digest.update(sim.full_brain.generator.get_state().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    parser.add_argument('--cycles', type=int, default=60)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('docs/results/coupling.json'))
    args = parser.parse_args()
    if min(args.cycles, args.trials) < 1: parser.error('Positive cycles and trials required')
    # Match the workbench server's CPU tensor threading configuration.
    torch.set_num_threads(1)
    results = []
    modes = ['serial', 'delayed_serial', 'pipelined']
    for trial in range(args.trials):
        for mode in (modes if trial % 2 == 0 else modes[::-1]):
            sim = load_live(args.checkpoint, args.graph)
            sim.full_brain.set_device('mps', 'float16')
            sim.set_coupling('serial' if mode == 'serial' else 'pipelined')
            with patch.object(module, 'run_pair', lambda brain, body: run_pair(brain, body, concurrent=mode != 'delayed_serial')):
                sim.advance(); sim.snapshot()
                torch.mps.synchronize()
                start_time = sim.time
                durations = []
                for _ in range(args.cycles):
                    started = time.perf_counter()
                    sim.advance(); sim.snapshot()
                    torch.mps.synchronize()
                    durations.append(time.perf_counter() - started)
            assert all(torch.isfinite(getattr(sim.full_brain, k)).all() for k in NEURAL_TENSORS)
            row = dict(trial=trial, mode=mode, cycles=args.cycles,
                       start_simulated_seconds=start_time, simulated_seconds=sim.time-start_time,
                       wall_seconds=sum(durations), mean_cycle_ms=1000*statistics.mean(durations),
                       cycles_per_wall_second=args.cycles/sum(durations),
                       endpoint_sha256=endpoint(sim), graph_sha256=sim.full_brain.manifest['graph_sha256'],
                       neurons=len(sim.full_brain.ids), edges=sim.full_brain.manifest['edges'],
                       execution=sim.full_brain.execution_summary(), command_hz=sim.command_hz,
                       scene=sim.body.scene_id, coupling=sim.coupling_summary(), cycle_seconds=durations)
            results.append(row)
            print(json.dumps({k: row[k] for k in ['trial','mode','wall_seconds','mean_cycle_ms','endpoint_sha256']}), flush=True)
            del sim; gc.collect(); torch.mps.empty_cache()
    delayed = [r['endpoint_sha256'] for r in results if r['mode'] != 'serial']
    assert len(set(delayed)) == 1, 'Parallel and same-delay serial endpoints differ'
    medians = {m: statistics.median(r['wall_seconds'] for r in results if r['mode']==m) for m in modes}
    report = dict(scope='Complete FP16 cycles with snapshots, one untimed warm-up cycle. Excludes loading, HTTP and browser. Serial has no added delay; delayed_serial and pipelined have identical one-cycle delay.',
                  torch_version=torch.__version__, mujoco_version=mujoco.__version__,
                  cpu_tensor_threads=torch.get_num_threads(),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  trials=results, exact_delayed_endpoints=True, median_wall_seconds=medians,
                  scheduling_speedup=medians['delayed_serial']/medians['pipelined'],
                  speedup_vs_zero_delay_serial=medians['serial']/medians['pipelined'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'}, indent=2), flush=True)


if __name__=='__main__': main()
