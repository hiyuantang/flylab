"""Probe full-history surrogate direction without modifying live or saved weights.

Uses an existing checkpoint configuration and the original adapter. The 500 ms
muscle target is physically validated before the diagnostic begins.
"""
import json,time,numpy as np,torch
from pathlib import Path
from threading import Event
from flylab.gesture_batch import BatchSession,demonstrations
from flylab.gesture_scene import random_placement
from flylab.gesture_history import clip_full_gradient
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
s=BatchSession(Path('data/full'),saved['settings'],2,42)
targets,evidence=demonstrations(s,25,Event(),cues=['point_both'])
place=random_placement(np.random.default_rng(42));e=s.gradient
base=e.parameters.detach().clone();t=time.perf_counter()
result,frame,comparison=s.sample('point_both',place,targets['point_both'],Event(),1.,train=True)
norm,log=clip_full_gradient(e);direction=e.parameters.grad.detach().clone()
energy=direction.square().sum(dim=(0,2)).cpu().numpy();top=np.argsort(energy)[-8:][::-1]
report={'steps':25,'command_ms':20,'backprop_window_ms':500,'original_adapter':True,'loss':result['loss'],'gradient_log10_norm':log,'gradient_seconds':time.perf_counter()-t,'top_gradient_groups':[{'group':e.adapter.groups[i],'squared_norm_share':float(energy[i]/energy.sum())} for i in top],'muscles':comparison,'probes':[]}
print(json.dumps({k:v for k,v in report.items() if k not in ['muscles']}),flush=True)
for amount in [-.1,-.01,-.001,0.,.001,.01,.1]:
 with torch.no_grad():e.parameters.copy_(base+amount*direction)
 r,_,_=s.sample('point_both',place,targets['point_both'],Event(),0.,train=False)
 row={'direction_displacement':amount,'loss':r['loss']};report['probes'].append(row);print(json.dumps(row),flush=True)
with torch.no_grad():e.parameters.copy_(base)
retinal=torch.cat([g[0] for g in s.sim.bridge.retina().groups]);prepare=s.sim.bridge.prepare_command
for cue in ['palm','point_both']:
 def blank(*a,**kw):
  d,m=prepare(*a,**kw);d[retinal]=0;return d,m
 s.sim.bridge.prepare_command=blank if cue=='point_both' else prepare
 r,_,c=s.sample(cue,place,targets['point_both'],Event(),0.,train=False)
 report['blank_vision' if cue=='point_both' else 'palm_same_target']={'loss':r['loss'],'activation_max_difference':float(np.max(np.abs(np.array(c['actual'])-np.array(comparison['actual']))))}
args.output.write_text(json.dumps(report,indent=2)+'\n')
