"""Isolated serial/parallel GPU comparison; never touches live or saved weights."""
import argparse
from dataclasses import asdict
from pathlib import Path
from threading import Event
import json
import time
import numpy as np
import torch
from flylab.body import BodyParameters, FlyBody
from flylab.gesture_batch import BatchSession, demonstrations
from flylab.gesture_scene import random_placement
from flylab.gesture_parallel import ParallelMetalBrain
from flylab.motor_mapping import EXTENDED_PROFILE
from flylab.senses import SensorySettings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=3)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--settings-from',type=Path)
    args=parser.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU required')
    settings={'body':asdict(BodyParameters(appendage_model='peripheral-v2',elasticity_profile='stance-elastic-v1')),'mapping_profile':EXTENDED_PROFILE,'gains':[0.]*8,
        'senses':asdict(SensorySettings(vision_enabled=True,vision_model='compound-retina-v1')),
        'training_execution':{'device':'mps','precision':'float16','gradient_precision':'float32'}}
    if args.settings_from:
        settings=torch.load(args.settings_from,map_location='cpu',weights_only=True)['settings']
    session=BatchSession(Path('data/full'),settings,2,42)
    targets,_=demonstrations(session,args.steps,Event())
    e=session.gradient
    cues=[('point','point_right','point_both')[i%3] for i in range(args.batch_size)]
    rng=np.random.default_rng(42)
    placements=[random_placement(rng) for _ in cues]
    results={}
    gradients={}
    samples={}
    for mode in ('sequential','parallel'):
        e.parameters.grad=None
        # Allocate and compile before timing. All graph buffers remain shared.
        if mode=='parallel':
            session.sim.reset();e.reset();e.sync_weights()
            session._parallel=ParallelMetalBrain(e,args.batch_size)
            session._parallel_bodies=[FlyBody(
                session.sim.body.parameters,session.sim.body.scene_id) for _ in range(args.batch_size-1)]+[session.sim.body]
            session._parallel_size=args.batch_size
            lib=session._parallel.lib
            peak={'tensor_bytes':0,'driver_bytes':0}
            class Probe:
                def __getattr__(self,name):return getattr(lib,name)
                def reduce_factors(self,*a,**kw):
                    lib.reduce_factors(*a,**kw)
                    if not peak['tensor_bytes']:
                        peak.update(tensor_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
            session._parallel.lib=Probe()
        torch.mps.synchronize()
        began=time.perf_counter()
        if mode=='sequential':
            outcomes=[]
            for cue,placement in zip(cues,placements):
                result,_,_=session.sample(cue,placement,targets[cue],Event(),1/len(cues))
                outcomes.append(result)
                print(mode,cue,'finished',flush=True)
        else:
            outcomes,_,_=session.parallel_samples(cues,placements,targets,Event(),
                progress=lambda **v:print(mode,v,flush=True))
        torch.mps.synchronize()
        from flylab.gesture_history import clip_full_gradient
        norm, log_norm = clip_full_gradient(e)
        results[mode]={'seconds':time.perf_counter()-began,'loss':float(np.mean([x['loss'] for x in outcomes])),
            'gradient_norm':norm, 'gradient_log10_norm':log_norm}
        if mode=='parallel':results[mode].update(peak)
        gradients[mode]=e.parameters.grad.cpu().clone()
        samples[mode]=outcomes
    a,b=gradients['sequential'],gradients['parallel']
    report={'batch_size':args.batch_size,'steps':args.steps,'precision':'FP16 neural / FP32 gradients',
        'neurons':len(session.sim.full_brain.ids),'edges':session.sim.full_brain.weights._nnz(),
        'results':results,'speedup':results['sequential']['seconds']/results['parallel']['seconds'],
        'max_clipped_gradient_difference':float((a-b).abs().max()),
        'gradient_relative_difference':float((a-b).norm()/a.norm().clamp_min(1e-12)),
        'max_final_joint_difference':float(np.max(np.abs(np.array([x['final_joints'] for x in samples['sequential']])-np.array([x['final_joints'] for x in samples['parallel']])))),
        'samples':samples}
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2),flush=True)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
