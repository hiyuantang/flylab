from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import pytest
from flylab.body import FlyBody, BodyParameters
from flylab.peripheral_mechanics import MUSCLES, PROFILE, MODEL, CHANNEL_COUNT
from flylab.neuromuscular import NeuromuscularBridge
from flylab.pretarsus import migrate_body
from flylab.live_state import save_live, load_live, NEURAL_TENSORS

PARAMETERS = BodyParameters(appendage_model=MODEL)


def test_each_named_channel_receives_its_own_neuron_and_generates_physical_force():
    rows = [{'bodyId': i+1, 'type': m.neuron_type, 'superclass': 'cb_motor' if m.subclass=='pm' else 'vnc_motor',
             'subclass': m.subclass, 'class': None, 'somaSide': m.side, 'instance': m.neuron_type+'_'+m.side}
            for i,m in enumerate(MUSCLES)]
    brain = SimpleNamespace(neurons=rows, ids=np.arange(len(rows)), voltage=torch.zeros(len(rows)), rates=torch.zeros(len(rows)))
    bridge = NeuromuscularBridge(brain, PROFILE)
    body = FlyBody(PARAMETERS)
    m,d = body.model,body.data
    assert m.nu == CHANNEL_COUNT == len(bridge.channels)
    for i, muscle in enumerate(MUSCLES):
        brain.rates.zero_();brain.rates[i]=100
        action=bridge.muscles()
        assert np.flatnonzero(action).tolist()==[90+i]
        assert m.actuator(90+i).name==muscle.name
        d.act[:]=0;mujoco.mj_forward(m,d);before=d.qfrc_actuator.copy()
        d.act[90+i]=1;mujoco.mj_forward(m,d)
        assert d.actuator_force[90+i] < 0
        for name, coefficient in muscle.joints:
            j=m.joint(name).dofadr[0]
            assert (d.qfrc_actuator[j]-before[j])*coefficient < 0
    assert bridge.summary()['mapped_motor_neurons']==len(MUSCLES)


def test_unknown_targets_do_not_get_fabricated_outputs():
    from flylab.motor_mapping import resolve_motor
    for name,subclass in [('MNwm36','wm'),('MNwm35','wm'),('MNad21','ad'),('MNnm13','nm'),('MNx01','pm'),('FNM2','nm'),('Fe reductor MN','fl')]:
        result=resolve_motor({'bodyId':1,'type':name,'subclass':subclass,'somaSide':'L','superclass':'vnc_motor'},PROFILE)
        assert result['status']=='unmapped'
    result=resolve_motor({'bodyId':1,'type':'MN9','subclass':'pm','somaSide':'L','instance':'MN9_R','superclass':'cb_motor'},PROFILE)
    assert result['reason']=='annotation_conflict'


def test_force_driven_proboscis_and_elastic_internal_walls():
    body=FlyBody(PARAMETERS);m,d=body.model,body.data
    # Lift the animal clear of the floor during the short muscle pulse.
    d.qpos[2]+=5
    action=np.zeros(m.nu)
    for name in ['l_m9','r_m9','l_m4a','r_m4a','l_pharyngeal_11d','r_pharyngeal_11d']:
        action[m.actuator(name).id]=1
    body.step_muscles(action,.01)
    assert d.qpos[m.joint('c_rostrum_protraction').qposadr][0]>.005
    assert d.qpos[m.joint('c_haustellum_extension').qposadr][0]>.005
    joint=m.joint('c_pharyngeal_11d_dilation').qposadr[0]
    dilated=d.qpos[joint];assert 0<dilated<.027
    body.step_muscles(np.zeros(m.nu),.03)
    assert d.qpos[joint]<dilated
    assert np.isfinite(d.qpos).all()


def test_migration_keeps_original_segment_poses_and_complete_neural_continuation(tmp_path):
    from test_full_connectome import graph_fixture
    from flylab.simulation import Simulation
    from flylab.full_connectome import Physiology
    from flylab.motor_mapping import PRETARSAL_PROFILE
    graph_fixture(tmp_path);sim=Simulation()
    sim.configure('connectome', tmp_path/'full', Physiology.paper(), BodyParameters(appendage_model='pretarsal-v1'),mapping_profile=PRETARSAL_PROFILE)
    sim.bridge.set_parameters(np.linspace(-.1,.1,8));sim.advance()
    old=sim.body; neural={name:getattr(sim.full_brain,name).clone() for name in NEURAL_TENSORS}
    sim.body=migrate_body(old,FlyBody(PARAMETERS))
    for name,index in old.body_ids.items():
        np.testing.assert_allclose(old.data.xpos[index],sim.body.data.xpos[sim.body.body_ids[name]],rtol=0,atol=1e-12)
    gains=sim.full_brain.output_gain
    values=sim.bridge.parameters.copy()
    sim.bridge=NeuromuscularBridge(sim.full_brain,PROFILE)
    sim.bridge.parameters=values;sim.full_brain.output_gain=gains
    for name,value in neural.items():torch.testing.assert_close(getattr(sim.full_brain,name),value,rtol=0,atol=0)
    path=tmp_path/'live.pt';save_live(sim,path);restored=load_live(path,tmp_path/'full')
    sim.advance();restored.advance()
    for name in NEURAL_TENSORS:torch.testing.assert_close(getattr(sim.full_brain,name),getattr(restored.full_brain,name),rtol=0,atol=0)
    for name in ['qpos','qvel','act','qacc_warmstart']:
        np.testing.assert_array_equal(getattr(sim.body.data,name),getattr(restored.body.data,name))
    assert len(sim.body.snapshot()['peripheral_muscles'])==len(MUSCLES)


def test_new_body_muscle_rl_shape_and_supported_scene_resets():
    from flylab.env import FlyEnv
    from flylab.scenes import SCENE_IDS
    env=FlyEnv(action_mode='muscle',body_parameters=PARAMETERS)
    obs,_=env.reset(seed=1)
    assert env.action_space.shape==(CHANNEL_COUNT,)
    obs,*_=env.step(np.zeros(CHANNEL_COUNT,dtype=np.float32))
    assert env.observation_space.contains(obs)
    for scene in SCENE_IDS:
        b=FlyBody(PARAMETERS,scene)
        b.step_muscles(np.zeros(CHANNEL_COUNT),.001)
        assert np.isfinite(b.data.qpos).all()
