"""Bounded MJX/Metal trial of an exported, unchanged MuJoCo fly model/state.

Run from FlyLab with --python pointing to an isolated MJX environment. Compilation
and unsupported-feature errors are recorded separately from warmed step timing.
"""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, help='Python in isolated MJX environment')
    parser.add_argument('--checkpoint', type=Path, default=Path('data/live-state.pt'))
    args = parser.parse_args()
    code='''
import json,time,sys
import numpy as np
import jax
import mujoco
from mujoco import mjx
print('devices',jax.devices(),flush=True)
m=mujoco.MjModel.from_binary_path(sys.argv[1])
d=mujoco.MjData(m)
state=np.load(sys.argv[2])
spec=mujoco.mjtState.mjSTATE_INTEGRATION
mujoco.mj_setState(m,d,state,spec);mujoco.mj_forward(m,d);mujoco.mj_setState(m,d,state,spec)
print(json.dumps(dict(nv=m.nv,nu=m.nu,ngeom=m.ngeom,ntendon=m.ntendon,timestep=m.opt.timestep)),flush=True)
print('converting_model',flush=True)
mx=mjx.put_model(m,impl='jax')
dx=mjx.put_data(m,d,impl='jax')
print('converted_model',dx.qpos.device,flush=True)
assert dx.qpos.device.platform.lower()=='metal'
print('compiling_full_fly_step',flush=True)
start=time.perf_counter()
step=jax.jit(mjx.step)
y=step(mx,dx)
y.qpos.block_until_ready()
print('compile_and_first_step_seconds',time.perf_counter()-start,flush=True)
mujoco.mj_step(m,d)
print('first_step_max_qpos_error',np.max(np.abs(np.asarray(y.qpos)-d.qpos)),flush=True)
start=time.perf_counter()
for _ in range(10):y=step(mx,y)
y.qpos.block_until_ready()
print('mean_warm_step_seconds',(time.perf_counter()-start)/10,flush=True)
print('finite',np.isfinite(np.asarray(y.qpos)).all(),flush=True)
'''
    import torch
    import mujoco
    import numpy as np
    from flylab.body import FlyBody, BodyParameters
    from flylab.live_state import model_fingerprint
    saved = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    body = FlyBody(BodyParameters(**saved['body_parameters']), saved['environment']['scene_id'])
    if model_fingerprint(body) != saved['body_sha256']:
        raise ValueError('Saved body differs from installed model')
    env = {**os.environ, 'JAX_PLATFORMS': 'METAL', 'ENABLE_PJRT_COMPATIBILITY': '1'}
    with tempfile.TemporaryDirectory(prefix='flylab-mjx-body-') as directory:
        model, state = Path(directory) / 'body.mjb', Path(directory) / 'state.npy'
        mujoco.mj_saveModel(body.model, str(model), None)
        np.save(state, saved['integration_state'].numpy())
        try:
            r = subprocess.run([args.python, '-c', code, str(model), str(state)],
                               capture_output=True, text=True, env=env, timeout=60)
            result = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr, timeout=False)
        except subprocess.TimeoutExpired as e:
            def decode(v): return v.decode(errors='replace') if isinstance(v, bytes) else v or ''
            result = dict(returncode=None, stdout=decode(e.stdout), stderr=decode(e.stderr), timeout=True)
    result['scope']='Unchanged full fly body/state, forced Metal. A timeout is not a speed measurement. Live neural simulation and dependencies unchanged.'
    Path('docs/results/mjx-body-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
