"""Experimental force-driven wing rig using MuJoCo's native ellipsoid fluid model.

Units: mm, g, s; forces are microNewtons and torques microNewton-mm. This
versioned engineering test environment does not modify the walking body or its
saved policies. Its bounded wing servos are not asynchronous flight muscles.
"""
from dataclasses import dataclass, asdict
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .body import BodyParameters, FlyBody, make_xml, vector

VERSION = 'wing-fluid-rig-v2'
WING_JOINTS = tuple(f'{s}_wing_{axis}' for s in ('l', 'r')
                    for axis in ('elevation', 'spread', 'feather'))
# Trim of wing kinematics in this model, not a fitted air density or body mass.
TRIM_FREQUENCY = 275.
TRIM_AMPLITUDE = 1.2033435049692482
TRIM_ELEVATION = .2858374930936211
TRIM_STROKE_BIAS = -.25
TRIM_PLANE_PITCH = .07913470832840196


@dataclass(frozen=True)
class FlightSettings:
    air_density: float = 1.225e-6  # 1.225 kg/m^3 -> g/mm^3
    air_viscosity: float = 1.81e-5  # kg/(m s) == g/(mm s)
    wind: tuple[float, float, float] = (0., 0., 0.)  # mm/s
    timestep: float = 0.00005  # Resolve a 200 Hz beat with 100 physics steps.
    fluid_coefficients: tuple[float, ...] = (1., .5, 1.5, 1.7, 1.)
    servo_stiffness: float = 100.  # uN mm/rad, engineering assumption
    servo_damping: float = .01  # uN mm s/rad
    torque_limit: float = 30.  # uN mm
    tethered: bool = True
    initial_height: float = 10.  # mm; release tests start clear of the floor
    stroke_plane_pitch: float = TRIM_PLANE_PITCH

    def __post_init__(self):
        values = [self.air_density, self.air_viscosity, *self.wind, self.timestep,
                  *self.fluid_coefficients, self.servo_stiffness, self.servo_damping,
                  self.torque_limit, self.initial_height, self.stroke_plane_pitch]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Flight settings must be finite')
        if len(self.wind) != 3 or len(self.fluid_coefficients) != 5:
            raise ValueError('Expected three wind components and five fluid coefficients')
        if (min(self.air_density, self.air_viscosity, *self.fluid_coefficients) < 0
                or not 1e-5 <= self.timestep <= 5e-5
                or min(self.servo_stiffness, self.torque_limit, self.initial_height) <= 0
                or self.servo_damping < 0 or abs(self.stroke_plane_pitch) > .3):
            raise ValueError('Invalid flight parameter range')


def flight_xml(settings, parameters):
    root = ET.fromstring(make_xml(parameters, 'lab'))
    root.set('model', VERSION)
    root.find('option').attrib.update(timestep=str(settings.timestep),
        density=str(settings.air_density), viscosity=str(settings.air_viscosity),
        wind=vector(settings.wind))
    bodies = {b.get('name'): b for b in root.iter('body')}
    # Flight axes differ from the old folded-wing positioning approximation.
    # The source meshes extend along +/-Y; X is chord and Z is wing-normal.
    for side, sign in [('l', 1), ('r', -1)]:
        wing = bodies[f'{side}_wing']
        wing.set('quat', vector([math.cos(settings.stroke_plane_pitch/2), 0,
                                 math.sin(settings.stroke_plane_pitch/2), 0]))
        # Stroke must be the outer rotation. Putting the large stroke angle
        # between elevation and feathering approaches an Euler singularity
        # near 90 degrees and couples their servo torques catastrophically.
        joints = {axis: wing.find(f"joint[@name='{side}_wing_{axis}']")
                  for axis in ('spread', 'elevation', 'feather')}
        for joint in joints.values():
            wing.remove(joint)
        for joint in joints.values():
            wing.append(joint)
        for axis, direction, limits in [('elevation', [sign, 0, 0], [-.8, .8]),
                                         ('spread', [0, 0, sign], [-1.7, 1.7]),
                                         ('feather', [0, 1, 0], [-1.7, 1.7])]:
            joint = wing.find(f"joint[@name='{side}_wing_{axis}']")
            joint.attrib.update(axis=vector(direction), range=vector(limits),
                                stiffness='0', damping='.00005', armature='1e-9')
            ET.SubElement(root.find('actuator'), 'position', name=f'flight_{side}_{axis}',
                joint=joint.get('name'), kp=str(settings.servo_stiffness), kv=str(settings.servo_damping),
                ctrlrange=vector(limits), forcerange=vector([-settings.torque_limit, settings.torque_limit]))
        # One massless fluid proxy per wing: no fluid forces from collision patches.
        # Approximate source mesh extent: 2.4 mm span x 1.08 mm chord. The proxy's
        # projected area is pi*1.2*.54 = 2.04 mm^2; thickness is assumed 0.02 mm.
        ET.SubElement(wing, 'geom', name=f'{side}_wing_airfoil', type='ellipsoid',
            size='.54 1.2 .01', pos=f'-.137 {sign * 1.1} .1', mass='0',
            contype='0', conaffinity='0', group='4', rgba='0 .5 1 .2',
            fluidshape='ellipsoid', fluidcoef=vector(settings.fluid_coefficients))
    ET.SubElement(root.find('worldbody'), 'body', name='flight_anchor', mocap='true',
                  pos=f'0 0 {settings.initial_height}')
    equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality, 'weld', name='flight_tether', body1='c_thorax', body2='flight_anchor',
                  relpose='0 0 0 1 0 0 0', active=str(settings.tethered).lower(), solref='.0002 1')
    return ET.tostring(root, encoding='unicode')


class FlightRig:
    """Keep actual wing/body integration separate from a commanded beat pattern."""
    def __init__(self, settings=FlightSettings()):
        self.settings = settings
        parameters = BodyParameters(appendage_model='peripheral-v2', elasticity_profile='stance-elastic-v1')
        self.body = FlyBody(parameters, 'lab')
        self.model = mujoco.MjModel.from_xml_string(flight_xml(settings, parameters))
        self.data = mujoco.MjData(self.model)
        # The compiler's minimum mass also populates the fixed mocap fixture.
        # Only the thorax subtree is the animal; the tether fixture is excluded.
        self.mass_g = float(self.model.body_subtreemass[self.model.body('c_thorax').id])
        self.root_id = self.model.body('c_thorax').id
        self.wing_ids = {self.model.body(f'{s}_wing').id for s in ('l', 'r')}
        self.body.model, self.body.data = self.model, self.data
        self.joint_ids = np.array([self.model.joint(n).id for n in WING_JOINTS])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([self.model.actuator(f'flight_{s}_{a}').id for s in ('l', 'r')
                                      for a in ('elevation', 'spread', 'feather')])
        self.reset()

    def reset(self):
        self.body.reset()
        self.data.qpos[:3] = [0., 0., self.settings.initial_height]
        self.data.qpos[3:7] = [1., 0., 0., 0.]
        self.data.qpos[self.qadr] = 0.
        self.data.qvel[:] = 0.
        self.data.ctrl[:] = 0.
        self.data.mocap_pos[0] = self.data.qpos[:3]
        self.data.mocap_quat[0] = self.data.qpos[3:7]
        self.phase = 0.
        self.elapsed_wall = 0.
        self.force_time = 0.
        mujoco.mj_forward(self.model, self.data)

    def advance(self, angles, duration):
        """Command six bounded servo angles; never overwrite integrated positions."""
        angles = np.asarray(angles, dtype=float)
        steps = round(duration / self.settings.timestep)
        if (angles.shape != (6,) or not np.isfinite(angles).all() or steps < 1
                or not math.isclose(steps * self.settings.timestep, duration, abs_tol=1e-12)):
            raise ValueError('Expected six finite angles and an integer number of physics steps')
        limits = self.model.actuator_ctrlrange[self.actuator_ids]
        if np.any(angles < limits[:, 0]) or np.any(angles > limits[:, 1]):
            raise ValueError('Wing angle command exceeds its physical control range')
        self.data.ctrl[self.actuator_ids] = angles
        began = time.perf_counter()
        previous_time = float(self.data.time)
        previous_warnings = self.data.warning.number.copy()
        mujoco.mj_step(self.model, self.data, nstep=steps)
        self.elapsed_wall += time.perf_counter() - began
        self.force_time = previous_time + (steps - 1) * self.settings.timestep
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or not np.isfinite(self.data.qfrc_fluid).all()
                or not math.isclose(self.data.time, previous_time + duration, abs_tol=1e-10)
                or np.any(self.data.warning.number != previous_warnings)):
            raise RuntimeError('Flight mechanics failed a solver or finite-state check')

    def flap(self, duration=.04, frequency=TRIM_FREQUENCY, amplitude=TRIM_AMPLITUDE,
             feather=.8, asymmetry=0., elevation=TRIM_ELEVATION,
             stroke_bias=TRIM_STROKE_BIAS, publish=None):
        """Engineering test driver; sinusoidal commands, force-driven actual wings.

        This is an explicit wingbeat controller, not learned gesture inference.
        Pattern updates occur at every physics step, independent of render rate.
        """
        if not (all(math.isfinite(x) for x in [duration, frequency, amplitude, feather, asymmetry,
                                             elevation, stroke_bias])
                and 0 <= frequency <= 300 and 0 <= amplitude <= 1.4
                and abs(feather) <= 1.4 and abs(asymmetry) <= .2
                and abs(elevation) <= .8 and abs(stroke_bias) + amplitude*(1+abs(asymmetry)) <= 1.7):
            raise ValueError('Invalid wingbeat parameters')
        steps = round(duration / self.settings.timestep)
        if steps < 1 or not math.isclose(steps * self.settings.timestep, duration, abs_tol=1e-12):
            raise ValueError('Duration must contain a whole number of physics steps')
        samples = []
        for index in range(steps):
            self.phase += 2 * math.pi * frequency * self.settings.timestep
            stroke = amplitude * math.sin(self.phase)
            pitch = feather * math.tanh(3 * math.cos(self.phase))
            self.advance([elevation, stroke_bias + stroke * (1 + asymmetry), pitch,
                          elevation, stroke_bias + stroke * (1 - asymmetry), pitch], self.settings.timestep)
            sample = self.measure()
            samples.append(sample)
            if publish:
                publish(sample, index)
        return samples

    def measure(self):
        # Generalized force on the free root has world XYZ translation followed
        # by root-local rotational coordinates. Do not label it pure wing lift:
        # the native inertia-based fluid model also acts on non-wing bodies.
        rotation = self.data.xmat[self.root_id].reshape(3, 3)
        root_torque = rotation @ self.data.qfrc_fluid[3:6]
        force = self.data.qfrc_fluid[:3]
        torque = root_torque - np.cross(
            self.data.subtree_com[self.root_id] - self.data.xpos[self.root_id], force)
        # mj_step retains pre-integration force/kinematic fields. Transform the
        # force at that same state, and expose its timestamp. The displayed
        # orientation, like qpos/qvel, is the new integrated state.
        current_rotation = np.empty(9)
        mujoco.mju_quat2Mat(current_rotation, self.data.qpos[3:7])
        return dict(time=float(self.data.time), position_mm=self.data.qpos[:3].tolist(),
            velocity_mm_s=self.data.qvel[:3].tolist(), upright=float(current_rotation[8]),
            quaternion=self.data.qpos[3:7].tolist(), angular_velocity_rad_s=self.data.qvel[3:6].tolist(),
            angles_rad=self.data.qpos[self.qadr].tolist(),
            fluid_force_uN=force.tolist(), fluid_torque_uN_mm=torque.tolist(),
            fluid_root_torque_uN_mm=root_torque.tolist(), force_time_s=self.force_time,
            servo_torque_uN_mm=self.data.actuator_force[self.actuator_ids].tolist(),
            servo_power_uW=float(np.dot(self.data.actuator_force[self.actuator_ids],
                                       self.data.actuator_velocity[self.actuator_ids])) * .001,
            weight_uN=self.mass_g * 9810., tethered=bool(self.data.eq_active[0]),
            contacts=int(self.data.ncon),
            environment_contacts=int(sum(c.dist < 0 and any(self.model.geom_bodyid[g] == 0 for g in c.geom)
                                         for c in self.data.contact)),
            wing_contacts=int(sum(c.dist < 0 and any(self.model.geom_bodyid[g] in self.wing_ids for g in c.geom)
                                  for c in self.data.contact)))

    def description(self):
        return dict(version=VERSION, settings=asdict(self.settings), mujoco=mujoco.__version__,
            units='mm, g, s; microNewtons; microNewton-mm; microWatts',
            controller='Bounded engineering wing servos, not asynchronous muscles',
            fluid='Native MuJoCo ellipsoid wings + inertia-based body drag; no resolved wake or flexible wings',
            body_count=int(self.model.nbody-2), muscle_count=len(self.data.ctrl)-6)


def average_wingbeats(samples, frequency):
    """Average whole cycles in the latter half, including fractional edge steps.

    Forces are MuJoCo's pre-integration values held over each physics interval.
    A partial beat can create a large spurious mean moment even in a trimmed rig.
    """
    first, end = samples[0]['force_time_s'], samples[-1]['time']
    cycles = math.floor((end-first)*frequency/2 + 1e-8) if frequency > 0 else 0
    start = end-cycles/frequency if cycles else (first+end)/2
    weights = np.array([max(0., min(s['time'], end)-max(s['force_time_s'], start)) for s in samples])
    def mean(key):
        return np.average([s[key] for s in samples], weights=weights, axis=0)
    return dict(mean_force_uN=mean('fluid_force_uN').tolist(),
        mean_torque_uN_mm=mean('fluid_torque_uN_mm').tolist(),
        mean_servo_power_uW=float(mean('servo_power_uW')),
        averaging_interval_s=[start, end], averaging_cycles=cycles)
