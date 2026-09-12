"""Measured compound-eye directions and explicitly inferred connectome routing.

Optics: Zhao et al. 2025 microCT specimen (857 left, 852 right).
MaleCNS PR->column: dominant synapse-count evidence, never ID order.
Column->optical-template registration is NOT anatomically validated.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch

MODEL = 'compound-retina-v1'
ASSET = Path(__file__).parent / 'assets/retina/eye-directions.json'


@lru_cache(maxsize=1)
def eye_template():
    return json.loads(ASSET.read_text())


@lru_cache(maxsize=1)
def optical_geometry():
    """Seven quadrature rays per measured axis approximate an acceptance cone."""
    result = []
    for side in ['left', 'right']:
        axes = np.array(eye_template()[side])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        tangent = np.cross(axes, np.array([0., 0., 1.]))
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        other = np.cross(axes, tangent)
        # 2 degree ring, normalized Gaussian quadrature. Acceptance width assumed.
        ring = np.arange(6) * np.pi / 3
        around = np.cos(ring)[None,:,None]*tangent[:,None] + np.sin(ring)[None,:,None]*other[:,None]
        rays = np.concatenate([axes[:,None], np.cos(np.deg2rad(2))*axes[:,None] + np.sin(np.deg2rad(2))*around],axis=1)
        result.append((axes, rays.reshape(-1,3)))
    return tuple(result)


def integrate_facets(rgb):
    values = rgb.reshape(-1,7,3)
    return (values[:,0]*.4 + values[:,1:].sum(1)*.1).clip(0,1)


def receptor_channels(rgb):
    # RGB material reflectance supplies only coarse visible-band proxies. UV is
    # unavailable rather than inferred from blue. Values are not calibrated opsins.
    return {'R1-R6': rgb @ np.array([.05,.65,.30]),
            'R8p': rgb[:,2], 'R8y': rgb[:,1]}


class RetinalRouting:
    def __init__(self, brain):
        self.records, self.routes = [], []
        directory = getattr(brain, 'directory', None)
        if directory is None:
            self.summary = {'mapped': 0, 'unresolved': 0, 'status': 'No connectivity available'}
            return
        ptr = np.load(directory/'indptr.npy', mmap_mode='r')
        pre = np.load(directory/'indices.npy', mmap_mode='r')
        counts = np.load(directory/'counts.npy', mmap_mode='r')
        rows = brain.neurons
        visual = {i for i,r in enumerate(rows) if r.get('superclass')=='ol_sensory' and r.get('class')=='visual'}
        evidence = {i:{} for i in visual}
        columns = {'L':set(), 'R':set()}
        for j,r in enumerate(rows):
            h,k,side = r.get('assignedOlHex1'),r.get('assignedOlHex2'),r.get('somaSide')
            if h is None or k is None or side not in columns:
                continue
            col=(int(h),int(k));columns[side].add(col)
            for i,w in zip(pre[ptr[j]:ptr[j+1]],counts[ptr[j]:ptr[j+1]]):
                i=int(i)
                if i in visual and rows[i].get('rootSide')==side:
                    evidence[i][col]=evidence[i].get(col,0)+int(w)
        # Centre aligned shared hex axes only. No claim of a measured transform
        # between MaleCNS hex numbering and this other specimen's optical map.
        template=np.array(eye_template()['right_hex'])
        lookup={tuple(h):i for i,h in enumerate(template)}
        centres={s:np.rint(np.median(sorted(c),axis=0)).astype(int) if c else np.zeros(2,dtype=int) for s,c in columns.items()}
        right_axes=optical_geometry()[1][0]
        mirrored=right_axes*np.array([1,-1,1])
        left_match=(mirrored@optical_geometry()[0][0].T).argmax(1)
        for i in sorted(visual):
            row=rows[i]; ev=evidence[i]; record={'body_id':int(row['bodyId']),'type':row.get('type'),'side':row.get('rootSide')}
            if not ev:
                record['reason']='no_column_evidence'
            else:
                ordered=sorted(ev.items(),key=lambda x:(-x[1],x[0])); col,w=ordered[0]; support=w/sum(ev.values())
                record.update(column=list(col),support=support,column_synapses=w,total_column_synapses=sum(ev.values()))
                target=lookup.get(tuple(np.array(col)-centres[row['rootSide']]))
                if support < .5 or w < 5 or (len(ordered)>1 and ordered[1][1]==w):
                    record['reason']='ambiguous_column'
                elif target is None:
                    record['reason']='outside_registered_template'
                elif row.get('type') not in {'R1-R6','R8p','R8y'}:
                    record['reason']='spectral_channel_unavailable'
                else:
                    side=0 if row['rootSide']=='L' else 1
                    facet=int(left_match[target]) if side==0 else target
                    record.update(facet=facet,reason=None)
                    self.routes.append((i,side,row['type'],facet))
            self.records.append(record)
        from collections import Counter
        self.summary={'mapped':len(self.routes),'unresolved':len(visual)-len(self.routes),
                      'reasons':dict(Counter(r['reason'] for r in self.records if r['reason'])),
                      'registration':'Centre-aligned hex axes across specimens; orientation and peripheral correspondence unvalidated',
                      'column_method':'Dominant same-eye annotated postsynaptic column, >=5 synapses, >=50% support, no ties',
                      'spectral':'RGB-derived visible bands only. UV, polarization and unclear receptor classes receive no invented input.',
                      'template_counts':[857,852]}
        self.groups=[]
        for side in range(2):
            for channel in ['R1-R6','R8p','R8y']:
                group=[(i,f) for i,s,c,f in self.routes if s==side and c==channel]
                if group:
                    ids,facets=zip(*group)
                    self.groups.append((torch.tensor(ids),side,channel,np.array(facets)))

    def apply(self, drive, vision, gain):
        for ids, side, channel, facets in getattr(self,'groups',[]):
            values=np.asarray(vision['eyes'][side]['channels'][channel])[facets]
            drive[ids]=torch.as_tensor(values*gain,dtype=drive.dtype)
