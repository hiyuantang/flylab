from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
import mujoco
from flylab.body import FlyBody
from flylab.senses import SensorSuite, SensorySettings, sensory_observation
from flylab.retina import MODEL, optical_geometry, integrate_facets, RetinalRouting


def test_measured_directions_acceptance_cones_and_variable_eye_counts():
    for count,(axes,rays) in zip([857,852],optical_geometry()):
        assert axes.shape==(count,3) and rays.shape==(count*7,3)
        np.testing.assert_allclose(np.linalg.norm(rays,axis=1),1,atol=1e-9)
        np.testing.assert_array_equal(rays.reshape(-1,7,3)[:,0],axes)
        np.testing.assert_allclose(integrate_facets(np.ones((count*7,3))),1)
    assert optical_geometry()[0][0][:,1].mean()>0
    assert optical_geometry()[1][0][:,1].mean()<0


def test_spatial_scene_sampling_is_pose_attached_and_not_mean_pooled():
    body=FlyBody();settings=SensorySettings(vision_model=MODEL,vision_enabled=True,calibration_sphere=True)
    suite=SensorSuite(settings);before=body.data.qpos.copy()
    frame=suite.sample(body,[0,0,0]);eyes=frame['vision']['eyes']
    assert [e['count'] for e in eyes]==[857,852]
    assert sensory_observation(frame).shape==(1713,)
    assert np.std(eyes[0]['channels']['R1-R6'])>.01
    assert suite.sample(body,[0,0,0])==frame
    np.testing.assert_array_equal(body.data.qpos,before)
    dark=SensorSuite(replace(settings,illumination=0)).sample(body,[0,0,0])
    assert np.max(dark['vision']['eyes'][0]['channels']['R1-R6'])==0
    body.data.qpos[3:7]=[0,0,0,1];mujoco.mj_forward(body.model,body.data)
    rotated=suite.sample(body,[0,0,0])
    assert rotated['vision']['eyes'][0]['channels']['R1-R6']!=eyes[0]['channels']['R1-R6']
    json.dumps(frame,allow_nan=False)


def retinal_fixture(path):
    # Two spatially separate receptors plus a tied assignment, a UV receptor and
    # a receptor connected only across eyes. Edges retain their measured weights.
    rows=[{'bodyId':100+i,'superclass':'ol_sensory','class':'visual','type':'R7y' if i==3 else 'R1-R6','rootSide':'R'} for i in range(5)]
    rows += [{'bodyId':200+i,'superclass':'ol_intrinsic','type':'L1','somaSide':side,'assignedOlHex1':h,'assignedOlHex2':k} for i,(h,k,side) in enumerate([(-4,0,'R'),(4,0,'R'),(0,0,'L')])]
    pres=[0,2,3,1,2,4];counts=[10,5,10,10,5,20]
    np.save(path/'indptr.npy',np.array([0,0,0,0,0,0,3,5,6]))
    np.save(path/'indices.npy',np.array(pres));np.save(path/'counts.npy',np.array(counts))
    return SimpleNamespace(directory=path,neurons=rows)


def test_column_evidence_routes_local_signals_and_rejects_ambiguity(tmp_path):
    routing=RetinalRouting(retinal_fixture(tmp_path))
    assert routing.summary['mapped']==2
    a,b=routing.routes
    assert a[3]!=b[3]
    eyes=[{'channels':{'R1-R6':np.zeros(n).tolist()}} for n in [857,852]]
    eyes[a[1]]['channels'][a[2]][a[3]]=1
    drive=torch.zeros(8);routing.apply(drive,{'eyes':eyes},3)
    assert drive[a[0]]==3 and drive[b[0]]==0
    assert torch.count_nonzero(drive)==1
    assert {r['body_id']:r['reason'] for r in routing.records}[102]=='ambiguous_column'
    assert {r['body_id']:r['reason'] for r in routing.records}[103]=='spectral_channel_unavailable'
    assert {r['body_id']:r['reason'] for r in routing.records}[104]=='no_column_evidence'


def test_compound_observation_and_live_checkpoint_roundtrip(tmp_path):
    from flylab.env import FlyEnv
    from flylab.simulation import Simulation
    from flylab.full_connectome import Physiology
    from flylab.live_state import save_live,load_live,NEURAL_TENSORS
    from test_senses import sensory_graph
    settings=SensorySettings(vision_model=MODEL,vision_enabled=True)
    env=FlyEnv(sensory_settings=settings,max_steps=2)
    obs,_=env.reset(seed=1);assert env.observation_space.contains(obs)
    assert len(obs)==338+1713
    sensory_graph(tmp_path);sim=Simulation();sim.configure('connectome',tmp_path/'full',Physiology.paper())
    sim.sensors=SensorSuite(settings);sim.advance();save_live(sim,tmp_path/'live.pt')
    restored=load_live(tmp_path/'live.pt',tmp_path/'full')
    assert restored.sensors.settings.vision_model==MODEL
    sim.advance();restored.advance()
    for key in NEURAL_TENSORS:torch.testing.assert_close(getattr(sim.full_brain,key),getattr(restored.full_brain,key),rtol=0,atol=0)
    np.testing.assert_array_equal(sim.body.data.qpos,restored.body.data.qpos)
