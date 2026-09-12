"""Explicit distal long-tendon approximation, not reconstructed attachment geometry.

Named long-tendon muscles pull the unguitractor apparatus. Here one spatial
transmission per leg pulls a hinged paired claw, with passive elastic return.
Geometry, strength and elastic constants are uncalibrated; adhesion is absent.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

MODEL = 'pretarsal-v1'
# Capsule endpoints in the hinged foot frame (mm). Shared with the renderer.
CLAW_CAPSULES = [
    [0., side * .015, 0., .012, side * .015, -.018, .004]
    for side in (-1, 1)
] + [
    [.012, side * .015, -.018, -.004, side * .015, -.025, .003]
    for side in (-1, 1)
]


def add_pretarsi(root, bodies, legs, parameters):
    tendons = ET.SubElement(root, 'tendon')
    actuators = root.find('actuator')
    for leg in legs:
        prefix = leg.lower() + '_pretarsus'
        parent = bodies[leg.lower() + '_tarsus5']
        ET.SubElement(parent, 'site', name=prefix+'_anchor', pos='-.025 0 -.025', size='.002')
        body = ET.SubElement(parent, 'body', name=prefix, pos='0 0 -.055')
        ET.SubElement(body, 'joint', name=prefix+'_flexion', axis='0 1 0', range='0 1.1',
                      stiffness='.05', damping='.0005', armature='1e-8', springref='0')
        for index, capsule in enumerate(CLAW_CAPSULES):
            ET.SubElement(body, 'geom', name=f'{prefix}_claw_{index}', type='capsule',
                          fromto=' '.join(map(str, capsule[:6])), size=str(capsule[6]),
                          mass=str(.000001 * parameters.mass_scale), group='1', contype='2', conaffinity='3')
        ET.SubElement(body, 'site', name=prefix+'_insertion', pos='0 0 -.012', size='.002')
        tendon = ET.SubElement(tendons, 'spatial', name=prefix+'_tendon', width='.002')
        ET.SubElement(tendon, 'site', site=prefix+'_anchor')
        ET.SubElement(tendon, 'site', site=prefix+'_insertion')
        ET.SubElement(actuators, 'muscle', name=prefix+'_pull', tendon=prefix+'_tendon',
                      lengthrange='.02 .07', force=str(20 * parameters.strength_scale),
                      timeconst='.002 .004', fpmax='.00001')


def migrate_body(source, target):
    """Transfer named physical state; added joints start relaxed, clocks continue.

    This changes mechanics, so is a model migration, not an exact continuation.
    The caller must back up the old model checkpoint before committing it.
    """
    if source.scene_id != target.scene_id:
        raise ValueError('Body migration requires the same scene')
    old, new, a, b = source.model, target.model, source.data, target.data
    old_names = {old.joint(j).name for j in range(old.njnt)}
    for j in range(new.njnt):
        if new.joint(j).name not in old_names:
            b.qpos[new.jnt_qposadr[j]] = new.qpos_spring[new.jnt_qposadr[j]]
            b.qvel[new.jnt_dofadr[j]] = 0
    for j in range(old.njnt):
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j)
        k = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT, name)
        if k < 0:
            raise ValueError('Cannot remove a physical joint from a live animal')
        nq, nv = (7, 6) if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (1, 1)
        u, v = old.jnt_qposadr[j], new.jnt_qposadr[k]
        b.qpos[v:v+nq] = a.qpos[u:u+nq]
        u, v = old.jnt_dofadr[j], new.jnt_dofadr[k]
        for field in ('qvel', 'qacc_warmstart', 'qfrc_applied'):
            getattr(b, field)[v:v+nv] = getattr(a, field)[u:u+nv]
    for i in range(old.nu):
        k = new.actuator(mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_ACTUATOR, i)).id
        b.ctrl[k] = a.ctrl[i]
        b.act[new.actuator_actadr[k]] = a.act[old.actuator_actadr[i]]
    for name, index in source.body_ids.items():
        b.xfrc_applied[target.body_ids[name]] = a.xfrc_applied[index]
    b.time = a.time
    for field in ('phases', 'magnitudes', 'command', 'target'):
        setattr(target, field, getattr(source, field).copy())
    warmstart = b.qacc_warmstart.copy()
    mujoco.mj_forward(new, b)
    b.qacc_warmstart[:] = warmstart
    if not np.isfinite(b.qpos).all():
        raise ValueError('Invalid migrated physical state')
    return target
