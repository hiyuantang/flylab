"""MuJoCo schematic fly with six antagonistic muscle pairs.

Lengths and forces are model units, not calibrated fly physiology. Muscle
transmissions and geometry are synthetic. No kinematic replay drives the joints.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np

LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]


def make_xml():
    legs, tendons, actuators = [], [], []
    for i, name in enumerate(LEGS):
        side = 1 if i < 3 else -1
        row = i % 3
        x = [.30, 0, -.32][row]
        dx = [.26, 0, -.28][row]
        legs.append(f'''<body name="{name}_upper" pos="{x} {side*.2} 0">
          <joint name="{name}_hip" axis="0 1 0" range="-.8 .8" damping=".015" stiffness=".008"/>
          <geom type="capsule" fromto="0 0 0 {dx} {side*.43} -.24" size=".035" mass=".004"/>
          <body name="{name}_lower" pos="{dx} {side*.43} -.24">
            <joint name="{name}_knee" axis="1 0 0" range="-.45 .45" damping=".012" stiffness=".02"/>
            <geom type="capsule" fromto="0 0 0 {dx*.5} {side*.17} -.43" size=".022" mass=".002"/>
            <site name="{name}_foot" pos="{dx*.5} {side*.17} -.43" size=".025"/>
          </body>
        </body>''')
        for sign, suffix in [(1, "flexor"), (-1, "extensor")]:
            tendons.append(f'<fixed name="{name}_{suffix}"><joint joint="{name}_hip" coef="{sign}"/></fixed>')
            actuators.append(f'<muscle name="{name}_{suffix}" tendon="{name}_{suffix}" lengthrange="-.9 .9" force=".035" timeconst=".03 .06"/>')
    return f'''<mujoco model="FlyLab schematic fly">
      <compiler angle="radian"/>
      <option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast"/>
      <default><geom friction="1 .02 .002" condim="3" rgba=".42 .29 .17 1"/><joint limited="true"/></default>
      <worldbody>
        <geom name="ground" type="plane" size="20 20 .1"/>
        <body name="thorax" pos="0 0 .74">
          <freejoint/>
          <geom type="ellipsoid" size=".38 .23 .2" mass=".025"/>
          <body name="abdomen" pos="-.43 0 -.03"><geom type="ellipsoid" size=".43 .21 .19" mass=".012"/></body>
          <body name="head" pos=".39 0 .04"><geom type="ellipsoid" size=".21 .23 .19" mass=".008"/></body>
          {''.join(legs)}
        </body>
      </worldbody>
      <tendon>{''.join(tendons)}</tendon>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>'''


class FlyBody:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(make_xml())
        self.data = mujoco.MjData(self.model)
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, self.model.nbody)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, motor, dt=.02):
        self.data.ctrl[:] = np.clip(motor, 0, 1)
        for _ in range(round(dt/self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Body simulation became non-finite")

    def snapshot(self):
        return {
            "bodies": [{"name": name, "position": self.data.xpos[i+1].tolist(), "quaternion": self.data.xquat[i+1].tolist()} for i, name in enumerate(self.names)],
            "feet": self.data.site_xpos.tolist(),
            "joints": self.data.qpos[7:].tolist(),
            "position": self.data.qpos[:3].tolist(),
            "velocity": self.data.qvel[:3].tolist(),
            "speed": float(np.linalg.norm(self.data.qvel[:2])),
            "activation": self.data.act.tolist(),
            "force": np.abs(self.data.actuator_force).tolist(),
            "contacts": int(self.data.ncon),
        }
