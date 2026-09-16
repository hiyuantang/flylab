"""Bounded isolated JAX/Metal probe; run with the experimental environment's Python.

The small matrix must actually execute on Metal before attempting a physics port.
No CPU fallback, live model changes, or dependency modifications are performed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/results/mjx-metal-probe.json'))
    args=parser.parse_args()
    code='''
import importlib.metadata as md
import json
print(json.dumps({p:md.version(p) for p in ['jax','jaxlib','jax-metal','mujoco','mujoco-mjx']}),flush=True)
import jax
import jax.numpy as jnp
print('devices',jax.devices(),flush=True)
a=jnp.ones((32,32),dtype=jnp.float32)
r=jax.jit(lambda x:x@x)(a).block_until_ready()
print('matrix_device',r.device,flush=True)
assert r.device.platform.lower() in ['metal','gpu'],r.device
print('matrix_value',float(r[0,0]),flush=True)
from mujoco import mjx
print('mjx_import_ok',flush=True)
'''
    env={**os.environ,'JAX_PLATFORMS':'METAL','ENABLE_PJRT_COMPATIBILITY':'1'}
    try:
        r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=45)
        result={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'timeout':False}
    except subprocess.TimeoutExpired as e:
        def decode(v): return v.decode(errors='replace') if isinstance(v,bytes) else v or ''
        result={'returncode':None,'stdout':decode(e.stdout),'stderr':decode(e.stderr),'timeout':True}
    result['scope']='Official Apple JAX Metal plug-in plus matching-version MuJoCo MJX, in isolated temporary environment; forces Metal, no CPU fallback.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
