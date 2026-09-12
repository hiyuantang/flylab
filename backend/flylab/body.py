"""Connected NeuroMechFly anatomy with an explicit, assumed muscle controller.

Geometry, segment masses, and joint frames come from the pinned FlyGym rig.
Units are mm, g, s (force: microNewtons). The 84 fixed-moment-arm Hill actuators
and reference-tracking VNC controller are engineering assumptions, not a
reconstructed muscle map. MuJoCo integrates every free body and joint state.
"""
from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from .scenes import get_scene

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
ASSETS = Path(__file__).with_name("assets")
MESHES = ASSETS / "meshes"
if not MESHES.is_dir():  # Editable source checkout shares the frontend assets.
    MESHES = Path(__file__).resolve().parents[2] / "frontend/public/models"
ANATOMY = json.loads((ASSETS / "anatomy.json").read_text())
SEGMENTS = ANATOMY["segments"]
ACTIVE_DOF = [("coxa", "yaw"), ("coxa", "pitch"), ("coxa", "roll"),
              ("trochanterfemur", "pitch"), ("trochanterfemur", "roll"),
              ("tibia", "pitch"), ("tarsus1", "pitch")]
AXES = {"yaw": (1, 0, 0), "pitch": (0, 1, 0), "roll": (0, 0, 1)}
MOMENT_ARM = .05  # mm; effective transmission, not anatomical tendon routing
MAX_FORCE = 1400.  # microNewtons; assumed actuator strength


@dataclass(frozen=True)
class BodyParameters:
    mass_scale: float = 1.
    strength_scale: float = 1.
    friction: float = 1.
    limit_mode: str = 'baseline'
    appendage_model: str = 'baseline'

    def __post_init__(self):
        if not all(np.isfinite(x) and .05 <= x <= 4 for x in [self.mass_scale, self.strength_scale, self.friction]):
            raise ValueError('Body scales and friction must be finite and within [0.05, 4]')
        if self.appendage_model not in {'baseline', 'pretarsal-v1', 'peripheral-v1'}:
            raise ValueError('Unknown appendage model')
        if self.limit_mode not in {'baseline', 'reference_envelope'}:
            raise ValueError('Unknown joint limit mode')


def mesh_name(name):
    return "l" + name[1:] if name.startswith("r") else name


def joint_name(segment, axis):
    return f"{segment}_{axis}"


def neutral_angle(segment, axis):
    parent = SEGMENTS[segment]["parent"]
    key = f"{parent}-{segment}-{axis}"
    if segment.startswith("r"):
        key = key.replace("-r", "-l")
        if key.startswith("r"):
            key = "l" + key[1:]
    return ANATOMY["neutral_angles"].get(key, 0.)


def vector(values):
    return " ".join(str(float(x)) for x in values)


@lru_cache(maxsize=1)
def wing_collision_patches():
    """Cover the curved wing surface with small convex sections, in local mm.

    Assign whole triangles to sections, preserving coverage at section boundaries.
    A 5 µm shell thickness avoids degenerate, perfectly flat collision meshes.
    """
    raw = (MESHES / "l_wing.stl").read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    record = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(raw, dtype=record, count=count, offset=84)["vertices"].astype(float) * 1000
    centers = triangles.mean(axis=1)
    # The source wing's long axis is local Y and chord is approximately X.
    bins_y = np.clip(((centers[:, 1] - centers[:, 1].min()) / np.ptp(centers[:, 1]) * 6).astype(int), 0, 5)
    bins_x = (centers[:, 0] > np.median(centers[:, 0])).astype(int)
    patches = []
    for span in range(6):
        for chord in range(2):
            vertices = np.unique(triangles[(bins_y == span) & (bins_x == chord)].reshape(-1, 3), axis=0)
            if not len(vertices):
                continue
            vertices = np.concatenate([vertices + [0, 0, .0025], vertices - [0, 0, .0025]])
            patches.append((span, vertices))
    return patches


@lru_cache(maxsize=1)
def walking_reference():
    with np.load(ASSETS / "walking_reference.npz", allow_pickle=False) as data:
        return data["angles"].copy(), data["swing_end"].copy()


def reference_angles(phases):
    values, _ = walking_reference()
    f = np.mod(phases, 2 * np.pi) / (2 * np.pi) * (values.shape[-1] - 1)
    i = np.floor(f).astype(int)
    a = values[np.arange(6), :, i]
    b = values[np.arange(6), :, np.minimum(i + 1, values.shape[-1] - 1)]
    return a + (b - a) * (f - i)[:, None]


def make_xml(parameters=BodyParameters(), scene_id='lab'):
    scene = get_scene(scene_id)
    root = ET.Element("mujoco", model="FlyLab articulated NeuroMechFly")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true",
                  boundmass="1e-6", boundinertia="1e-12", fusestatic="false")
    ET.SubElement(root, "option", timestep=".0001", gravity="0 0 -9810",
                  integrator="implicitfast", iterations="50", noslip_iterations="3")
    ET.SubElement(root, 'size', memory='64M')
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction=f"{parameters.friction} .005 .0001", condim="3",
                  solref=".0004 1", solimp=".999 .9999 .0001", margin=".002")
    assets = ET.SubElement(root, "asset")
    for name in SEGMENTS:
        ET.SubElement(assets, "mesh", name=name,
                      file=str(MESHES / f"{mesh_name(name)}.stl"),
                      scale=f"1000 {'-1000' if name.startswith('r') else '1000'} 1000")
        if name.endswith("wing"):
            for index, (_, vertices) in enumerate(wing_collision_patches()):
                vertices = vertices * [1, -1 if name.startswith("r") else 1, 1]
                ET.SubElement(assets, "mesh", name=f"{name}_surface_{index}", vertex=vector(vertices.ravel()))
    world = ET.SubElement(root, "worldbody")
    floor_shape = dict(type='plane', size='100 100 .1') if scene_id == 'lab' else dict(type='box', size=vector([scene['extent'][0] / 2, scene['extent'][1] / 2, 20]), pos='0 0 -20')
    rgb = lambda color: vector([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)] + [1])
    ET.SubElement(world, "geom", name="ground", **floor_shape, rgba=rgb(scene['floor_color']),
                  contype="1", conaffinity="2", priority="1",
                  solref=".002 1", solimp=".95 .99 .001")
    for obj in scene['objects']:
        ET.SubElement(world, 'geom', name=obj['id'], type=obj['shape'], pos=vector(obj['position']),
                      size=vector(obj['size']), quat=vector(obj['quaternion']), rgba=rgb(obj['color']),
                      group='0', contype='1', conaffinity='2', priority='1', solref='.002 1', solimp='.95 .99 .001')
    bodies = {}
    pending = list(SEGMENTS)
    while pending:
        for name in pending[:]:
            cfg = SEGMENTS[name]
            parent = cfg["parent"]
            if parent is not None and parent not in bodies:
                continue
            orientation = np.array(cfg["quat"], dtype=float)
            position = list(cfg["pos"])
            if name.endswith("wing"):
                # Fit the hinge on the dorsal thorax so folded wings clear the
                # abdomen. This 0.25 mm lift and resting angle are assumptions.
                position[2] += .25
                tilt = np.array([math.cos(.15 / 2), 0, math.sin(.15 / 2), 0])
                mujoco.mju_mulQuat(orientation, tilt, orientation.copy())
                spread = -.6 if name.startswith("l") else .6
                rotation = np.array([math.cos(spread / 2), 0, 0, math.sin(spread / 2)])
                mujoco.mju_mulQuat(orientation, rotation, orientation.copy())
            body = ET.SubElement(world if parent is None else bodies[parent], "body",
                                 name=name, pos=vector([0, 0, 1.3] if parent is None else position),
                                 quat=vector(orientation))
            if parent is None:
                ET.SubElement(body, "freejoint", name="root")
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                          group="1",
                          mass=str(cfg["mass"] * parameters.mass_scale), contype="0" if name.endswith("wing") else "2",
                          conaffinity="0" if name.endswith("wing") else "3")
            if name.endswith("wing"):
                for index, _ in enumerate(wing_collision_patches()):
                    patch = f"{name}_surface_{index}"
                    ET.SubElement(body, "geom", name=patch, mesh=patch, type="mesh", mass="0",
                                  contype="2", conaffinity="3", group="3")
                inverse = orientation.copy()
                inverse[1:] *= -1
                axis = np.zeros(3)
                mujoco.mju_rotVecQuat(axis, np.array([0., 1., 0.]), inverse)
                ET.SubElement(body, "joint", name=f"{name}_elevation", type="hinge",
                              axis=vector(axis), range="-.1 .5", springref="0",
                              stiffness=".5", damping=".005", armature="1e-6",
                              solreflimit=".0004 1", solimplimit=".999 .9999 .0001")
                mujoco.mju_rotVecQuat(axis, np.array([0., 0., -1. if name.startswith("l") else 1.]), inverse)
                ET.SubElement(body, "joint", name=f"{name}_spread", type="hinge",
                              axis=vector(axis), range="-.08 .8", springref="0",
                              stiffness=".5", damping=".005", armature="1e-6",
                              solreflimit=".0004 1", solimplimit=".999 .9999 .0001")
            leg = name[:2].upper()
            if leg in LEGS:
                link = name.split("_", 1)[1]
                axes = {"coxa": ["yaw", "pitch", "roll"],
                        "trochanterfemur": ["pitch", "roll"]}.get(link, ["pitch"])
                passive = link in {"tarsus2", "tarsus3", "tarsus4", "tarsus5"}
                for axis in axes:
                    sign = -1 if name.startswith("r") and axis != "pitch" else 1
                    rest = neutral_angle(name, axis)
                    span = .7 if passive else 1.4
                    limits = [rest - span, rest + span]
                    if not passive and parameters.limit_mode == 'reference_envelope':
                        trajectory = walking_reference()[0][LEGS.index(leg), ACTIVE_DOF.index((link, axis))]
                        limits = [min(rest, float(trajectory.min())) - .35, max(rest, float(trajectory.max())) + .35]
                    ET.SubElement(body, "joint", name=joint_name(name, axis), type="hinge",
                                  axis=vector(np.array(AXES[axis]) * sign),
                                  range=vector(limits),
                                  springref=str(rest), stiffness="7.5" if passive else ".05",
                                  damping=".01" if passive else ".06", armature="1e-6")
                if link == "tarsus5":
                    ET.SubElement(body, "site", name=f"{leg}_foot", pos="0 0 -.085", size=".025")
            bodies[name] = body
            pending.remove(name)
    # The broad phase checks nonadjacent segments, including legs against the
    # body and other legs. Adjacent joint sockets are intentionally excluded.
    contact = ET.SubElement(root, "contact")
    # MuJoCo's parent-weld filter would otherwise exclude these wing/body
    # contacts along with the intended shoulder attachment exclusion.
    for wing in ("l_wing", "r_wing"):
        for index, (span, _) in enumerate(wing_collision_patches()):
            for other in SEGMENTS:
                if other.startswith("c_abdomen") or other == "c_head" or (other == "c_thorax" and span > 0):
                    ET.SubElement(contact, "pair", geom1=f"{wing}_surface_{index}", geom2=other,
                                  solref=".0004 1", solimp=".999 .9999 .0001")
    for leg in LEGS:
        for other in SEGMENTS:
            if other.startswith("c_abdomen") or other == "c_head":
                ET.SubElement(contact, "pair", geom1=f"{leg.lower()}_coxa", geom2=other,
                              solref=".0004 1", solimp=".999 .9999 .0001")
    actuators = ET.SubElement(root, "actuator")
    for leg in LEGS:
        for link, axis in ACTIVE_DOF:
            name = joint_name(f"{leg.lower()}_{link}", axis)
            for sign, suffix in [(-1, "positive"), (1, "negative")]:
                ET.SubElement(actuators, "muscle", name=f"{name}_{suffix}", joint=name,
                              gear=str(sign * MOMENT_ARM), lengthrange="-.4 .4",
                              force=str(MAX_FORCE * parameters.strength_scale), timeconst=".002 .004", fpmax=".00001")
    if parameters.appendage_model in {"pretarsal-v1", "peripheral-v1"}:
        from .pretarsus import add_pretarsi
        add_pretarsi(root, bodies, LEGS, parameters)
    if parameters.appendage_model == "peripheral-v1":
        from .peripheral_mechanics import add_mechanics
        add_mechanics(root, bodies, parameters)
    return ET.tostring(root, encoding="unicode")


@lru_cache(maxsize=8)
def compiled_model(parameters=BodyParameters(), scene_id='lab'):
    return mujoco.MjModel.from_xml_string(make_xml(parameters, scene_id))


class FlyBody:
    def __init__(self, parameters=BodyParameters(), scene_id='lab'):
        self.parameters = parameters
        self.scene_id = scene_id
        self.scene = get_scene(scene_id)
        self.model = compiled_model(parameters, scene_id)
        self.data = mujoco.MjData(self.model)
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                      for i in range(1, self.model.nbody)]
        self.body_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                         for name in self.names}
        self.active_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                   joint_name(f"{leg.lower()}_{link}", axis))
                                   for leg in LEGS for link, axis in ACTIVE_DOF])
        self.qadr = self.model.jnt_qposadr[self.active_ids]
        self.vadr = self.model.jnt_dofadr[self.active_ids]
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot") for leg in LEGS]
        self.rest = reference_angles(np.ones(6) * np.pi)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for name in self.names:
            for axis in AXES:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name(name, axis))
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = neutral_angle(name, axis)
        self.data.qpos[self.qadr] = self.rest.ravel()
        self.data.qpos[:3] += self.scene['spawn']
        mujoco.mj_forward(self.model, self.data)
        self._separate_reset_contacts()
        self.data.qpos[2] += self.scene['spawn'][2] + .03 - np.min(self.data.site_xpos[self.foot_ids, 2])
        self.phases = np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=float)
        self.magnitudes = np.zeros(6)
        self.command = np.zeros(12)
        self.target = self.rest.copy()
        mujoco.mj_forward(self.model, self.data)

    def _separate_reset_contacts(self):
        """Project the initial joint pose out of self-intersections before time starts.

        Only reset uses position correction. Running simulation uses contact forces.
        Joint origins stay attached and root position/orientation are unchanged.
        """
        m, d = self.model, self.data
        jac_a, jac_b = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        hinges = np.arange(1, m.njnt)
        for _ in range(80):
            rows, gaps = [], []
            for contact in d.contact:
                if self.model.geom_bodyid[contact.geom1] == 0 or self.model.geom_bodyid[contact.geom2] == 0 or contact.dist >= .001:
                    continue
                a, b = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
                mujoco.mj_jac(m, d, jac_a, None, contact.pos, a)
                mujoco.mj_jac(m, d, jac_b, None, contact.pos, b)
                row = contact.frame[:3] @ (jac_b - jac_a)
                row[:6] = 0
                if np.linalg.norm(row) > 1e-8:
                    rows.append(row)
                    gaps.append(.003 - contact.dist)
            if not rows:
                return
            matrix = np.asarray(rows)
            update = matrix.T @ np.linalg.solve(matrix @ matrix.T + .0001 * np.eye(len(rows)), np.asarray(gaps))
            update *= min(1., .15 / max(np.max(np.abs(update)), 1e-12))
            mujoco.mj_integratePos(m, d.qpos, update, 1.)
            addresses = m.jnt_qposadr[hinges]
            d.qpos[addresses] = np.clip(d.qpos[addresses], m.jnt_range[hinges, 0], m.jnt_range[hinges, 1])
            mujoco.mj_forward(m, d)
        raise RuntimeError("Could not find a nonintersecting initial joint pose")

    def step(self, motor, dt=.02):
        motor = np.asarray(motor, dtype=float)
        if motor.shape != (12,) or not np.isfinite(motor).all():
            raise ValueError("Expected 12 finite motor-group drives")
        self.command = np.clip(motor, 0, 1)
        enabled = np.max(self.command.reshape(6, 2), axis=1) > 1e-5
        drive = np.mean(self.command.reshape(6, 2), axis=1)
        h = self.model.opt.timestep
        # Recompute tracking commands at 1 kHz; integrate muscle/contact physics at 10 kHz.
        biases = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        for step in range(round(dt / h)):
            coupling = 4 * np.sin(self.phases[None, :] - self.phases[:, None] - (biases[None, :] - biases[:, None])).sum(axis=1)
            self.phases += (2 * np.pi * (5 + 9 * drive) + coupling) * enabled * h
            self.magnitudes += (np.clip(drive * 2, 0, 1.2) - self.magnitudes) * 20 * h
            if step % 10 == 0:
                self.target = self.rest + self.magnitudes[:, None] * (reference_angles(self.phases) - self.rest)
                torque = 45 * (self.target.ravel() - self.data.qpos[self.qadr]) - .025 * self.data.qvel[self.vadr]
                torque = np.clip(torque, -60, 60)
                # Hill muscle gain varies with length/velocity; use a bounded nominal
                # conversion. This feedback controller is explicit, not learned biology.
                activation = np.stack((np.maximum(torque, 0), np.maximum(-torque, 0)), axis=1) / (MAX_FORCE * self.parameters.strength_scale * MOMENT_ARM)
                activation = np.clip(activation + .025, 0, 1)
                activation *= np.repeat(enabled, 7)[:, None]
                self.data.ctrl[:] = 0
                self.data.ctrl[:84] = activation.ravel()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Body simulation became non-finite")

    def step_muscles(self, excitation, dt=.02):
        """Direct RL control: bypass the reference tracker and command all installed muscles."""
        excitation = np.asarray(excitation, dtype=float)
        if excitation.shape != (self.model.nu,) or not np.isfinite(excitation).all():
            raise ValueError(f"Expected {self.model.nu} finite muscle excitations")
        self.data.ctrl[:] = np.clip(excitation, 0, 1)
        for _ in range(round(dt / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Body simulation became non-finite")

    def step_posture(self, target, log_gains=None, dt=.02):
        """Engineering posture baseline: joint feedback, no walking oscillator."""
        gains = np.zeros(8) if log_gains is None else np.asarray(log_gains)
        target = np.asarray(target)
        if gains.shape != (8,) or target.shape != (42,) or not np.isfinite(gains).all() or not np.isfinite(target).all():
            raise ValueError('Expected finite 42-angle target and eight log gains')
        for _ in range(round(dt / .001)):
            torque = (45 * np.exp(gains[6]) * (target - self.data.qpos[self.qadr])
                      - .025 * np.exp(gains[7]) * self.data.qvel[self.vadr])
            torque *= np.repeat(np.exp(gains[:6]), 7)
            activation = np.stack((np.maximum(torque, 0), np.maximum(-torque, 0)), axis=1)
            activation = activation / (MAX_FORCE * self.parameters.strength_scale * MOMENT_ARM) + .025
            excitation = np.zeros(self.model.nu)
            excitation[:84] = np.clip(activation.ravel(), 0, 1)
            self.step_muscles(excitation, dt=.001)

    def actuated_joint_ids(self):
        ids = set(self.active_ids.tolist())
        for j in range(self.model.njnt):
            name = self.model.joint(j).name
            if name.endswith('_pretarsus_flexion'):
                ids.add(j)
        if self.parameters.appendage_model == 'peripheral-v1':
            from .peripheral_mechanics import MUSCLES
            ids.update(self.model.joint(name).id for muscle in MUSCLES for name, _ in muscle.joints)
        return ids

    def peripheral_snapshot(self):
        if self.parameters.appendage_model != 'peripheral-v1':
            return []
        from .peripheral_mechanics import MUSCLES
        return [{'name': m.name, 'target': m.target, 'region': m.subclass, 'side': m.side,
                 'activation': float(self.data.act[90+i]), 'force_uN': float(abs(self.data.actuator_force[90+i])),
                 'length_mm': float(self.data.actuator_length[90+i]), 'force_parameter_uN': m.force*self.parameters.strength_scale,
                 'joints': [n for n, _ in m.joints], 'interpretation': m.interpretation}
                for i, m in enumerate(MUSCLES)]

    def additional_geometry(self, muscles):
        return [{'name': name, 'position': self.data.xpos[index].tolist(),
                 'quaternion': self.data.xquat[index].tolist(), 'capsules': [[0, 0, 0, .055, 0, 0, .025]],
                 'activation': max((m['activation'] for m in muscles if m['side'].lower() == name[0] and m['target'] in {'m6', 'm7'}), default=0.)}
                for name, index in self.body_ids.items() if name.endswith('_labellum')]

    def mechanics(self):
        joints = []
        active = self.actuated_joint_ids()
        for jid in range(1, self.model.njnt):
            dof = self.model.jnt_dofadr[jid]
            joints.append({'name': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                           'range_degrees': np.rad2deg(self.model.jnt_range[jid]).tolist(),
                           'range': (self.model.jnt_range[jid] if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE else np.rad2deg(self.model.jnt_range[jid])).tolist(),
                           'range_unit': 'mm' if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE else '°',
                           'stiffness': float(self.model.jnt_stiffness[jid]),
                           'damping': float(self.model.dof_damping[dof]), 'active': bool(jid in active)})
        return {'parameters': asdict(self.parameters), 'mass_mg': float(self.model.body_mass.sum() * 1000),
                'max_force_parameter_uN': MAX_FORCE * self.parameters.strength_scale,
                'pretarsal_max_force_parameter_uN': 20 * self.parameters.strength_scale if self.model.nu >= 90 else None,
                'moment_arm_mm': MOMENT_ARM, 'joints': joints,
                'provenance': 'Main-body masses and geometry: pinned female FlyGym rig. Added pretarsal and peripheral geometry, masses, transmissions and forces are uncalibrated approximations. No adhesion, fluid transport, asynchronous flight muscles or aerodynamics. Strength, springs, damping and baseline limits: assumptions. Reference envelope: observed gait extrema plus 0.35 rad margin, not anatomical maximum range.'}

    def foot_feedback(self):
        forces = np.zeros(6)
        contact_force = np.zeros(6)
        # Contact indices are distinct from constraint addresses.
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.model.geom_bodyid[contact.geom1] != 0 and self.model.geom_bodyid[contact.geom2] != 0:
                continue
            other = contact.geom2 if self.model.geom_bodyid[contact.geom1] == 0 else contact.geom1
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or ""
            if name[:2].upper() in LEGS:
                mujoco.mj_contactForce(self.model, self.data, index, contact_force)
                forces[LEGS.index(name[:2].upper())] += max(0., contact_force[0])
        return forces

    def pretarsal_snapshot(self):
        from .pretarsus import CLAW_CAPSULES
        return [{"name": name, "position": self.data.xpos[index].tolist(),
                 "quaternion": self.data.xquat[index].tolist(), "capsules": CLAW_CAPSULES,
                 "activation": float(self.data.act[84 + LEGS.index(name[:2].upper())])}
                for name, index in self.body_ids.items() if name.endswith("_pretarsus")]

    def snapshot(self):
        forces = self.foot_feedback()
        peripheral = self.peripheral_snapshot()
        grouped_activation = self.data.act[:84].reshape(6, 7, 2).mean(axis=1)
        grouped_force = np.abs(self.data.actuator_force[:84]).reshape(6, 7, 2).sum(axis=1)
        rotation = self.data.xmat[self.body_ids["c_thorax"]].reshape(3, 3)
        return {
            "bodies": [{"name": name, "parent": SEGMENTS[name]["parent"], "mesh": mesh_name(name),
                        "mirror": name.startswith("r"), "position": self.data.xpos[self.body_ids[name]].tolist(),
                        "quaternion": self.data.xquat[self.body_ids[name]].tolist()} for name in self.names if name in SEGMENTS],
            "pretarsi": self.pretarsal_snapshot(),
            "additional_geometry": self.additional_geometry(peripheral), "peripheral_muscles": peripheral,
            "feet": self.data.site_xpos[self.foot_ids].tolist(),
            "foot_forces": forces.tolist(), "foot_contacts": (forces > .02).tolist(),
            "joints": self.data.qpos[7:].tolist(), "joint_velocities": self.data.qvel[6:].tolist(),
            "position": self.data.qpos[:3].tolist(), "velocity": self.data.qvel[:3].tolist(),
            "speed": float(np.linalg.norm(self.data.qvel[:2])),
            "activation": grouped_activation.ravel().tolist(), "force": grouped_force.ravel().tolist(),
            "muscle_activation": self.data.act.tolist(), "muscle_force": np.abs(self.data.actuator_force).tolist(),
            "contacts": int(self.data.ncon), "upright": float(rotation[2, 2]),
            "heading": float(math.atan2(rotation[1, 0], rotation[0, 0])),
            "anatomy": {"name": "NeuroMechFly articulated body", "segments": len(self.names),
                        "joints": self.model.njnt - 1, "actuated_joints": len(self.actuated_joint_ids()), "muscles": self.model.nu,
                        "mass_mg": float(self.model.body_mass.sum() * 1000), "length_unit": "mm",
                        "force_unit": "µN", "specimen": "female", "muscle_map": "assumed effective actuators"},
        }
