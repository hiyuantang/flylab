"""Compare Python vs native MuJoCo substep loops with identical controls/state."""
import json,time,statistics
from pathlib import Path
import mujoco
import numpy as np
from flylab.live_state import load_live, STATE_SPEC


def main():
    sim=load_live(Path('data/live-state.pt'),Path('data/full'))
    m,d=sim.body.model,sim.body.data
    h=m.opt.timestep
    steps=int((1/sim.command_hz)//h)
    remainder=1/sim.command_hz-steps*h
    state=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,state,STATE_SPEC)
    trials=[];endpoints={}
    for trial in range(4):
        for mode in (['python','native'] if trial%2==0 else ['native','python']):
            mujoco.mj_setState(m,d,state,STATE_SPEC)
            mujoco.mj_forward(m,d);mujoco.mj_setState(m,d,state,STATE_SPEC)
            start=time.perf_counter()
            for _ in range(60):
                if mode=='python':
                    for _ in range(steps):mujoco.mj_step(m,d)
                else:mujoco.mj_step(m,d,nstep=steps)
                if remainder>1e-12:
                    m.opt.timestep=remainder
                    try:mujoco.mj_step(m,d)
                    finally:m.opt.timestep=h
                mujoco.mj_forward(m,d)
            elapsed=time.perf_counter()-start
            endpoint=np.empty_like(state);mujoco.mj_getState(m,d,endpoint,STATE_SPEC)
            if endpoints: np.testing.assert_array_equal(endpoint,next(iter(endpoints.values())))
            endpoints[mode]=endpoint
            trials.append(dict(trial=trial,mode=mode,wall_seconds=elapsed))
            print(trials[-1],flush=True)
    result=dict(scope='Body only: 60 command intervals, same held excitation and initial full integration state; all fractional steps retained.',
                simulated_seconds=60/sim.command_hz,endpoint_exact=True,trials=trials,
                median_seconds={mode:statistics.median(t['wall_seconds'] for t in trials if t['mode']==mode) for mode in ['python','native']})
    Path('docs/results/physics-batch.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
