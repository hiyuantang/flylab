"""Isolated GPU matrix experiment against the complete CPU muscle readout."""
from pathlib import Path
import argparse
import json
import statistics
import time
import numpy as np
import torch
from flylab.live_state import load_live
from flylab.matrix_readout import MatrixMotorReadout


def timed(fn, repeats):
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    torch.mps.synchronize()
    return (time.perf_counter()-start)/repeats, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    parser.add_argument('--graph', type=Path, default=Path('data/full'))
    args=parser.parse_args()
    sim=load_live(args.checkpoint,args.graph)
    assert sim.full_brain.device.type=='mps'
    reference=sim.bridge.muscles
    matrix=MatrixMotorReadout(sim.bridge)
    # Exercise active neurons even when the saved state starts at rest.
    original=sim.full_brain.rates.clone()
    sim.full_brain.rates=torch.linspace(0,200,len(sim.full_brain.ids),device='mps',dtype=sim.full_brain.dtype)
    expected=reference();actual=matrix()
    error=float(np.max(np.abs(expected-actual)))
    np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-6)
    trials=[]
    for trial in range(5):
        for name,fn in ([('cpu',reference),('gpu_matrix',matrix)] if trial%2==0 else [('gpu_matrix',matrix),('cpu',reference)]):
            fn()
            duration,_=timed(fn,100)
            trials.append(dict(trial=trial,mode=name,mean_ms=1000*duration))
    sim.full_brain.rates=original
    # End-to-end paths each restart from the exact same saved state and one warm-up.
    from flylab.live_state import NEURAL_TENSORS
    import gc
    del sim,matrix,reference,fn
    gc.collect();torch.mps.empty_cache()
    full=[]
    for trial in range(2):
        for mode in (['cpu','gpu_matrix'] if trial%2==0 else ['gpu_matrix','cpu']):
            sim=load_live(args.checkpoint,args.graph)
            if mode=='gpu_matrix': sim.bridge.muscles=MatrixMotorReadout(sim.bridge)
            sim.advance();sim.snapshot()
            start=time.perf_counter()
            for _ in range(30): sim.advance();sim.snapshot()
            torch.mps.synchronize()
            elapsed=time.perf_counter()-start
            full.append(dict(trial=trial,mode=mode,cycles=30,simulated_seconds=30/sim.command_hz,wall_seconds=elapsed))
            print(full[-1],flush=True)
            assert all(torch.isfinite(getattr(sim.full_brain,k)).all() for k in NEURAL_TENSORS)
            del sim;gc.collect();torch.mps.empty_cache()
    result=dict(scope='Existing motor mapping: CPU float64 weighted means vs GPU float32 matrix; neural precision unchanged. Different rounding can alter physical trajectories.',
                max_action_abs_error=error,microbench_trials=trials,complete_cycle_trials=full)
    result['median_readout_ms']={m:statistics.median(t['mean_ms'] for t in trials if t['mode']==m) for m in ['cpu','gpu_matrix']}
    result['median_complete_cycle_wall_seconds']={m:statistics.median(t['wall_seconds'] for t in full if t['mode']==m) for m in ['cpu','gpu_matrix']}
    Path('docs/results/matrix-readout.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['microbench_trials','complete_cycle_trials']},indent=2),flush=True)


if __name__=='__main__': main()
