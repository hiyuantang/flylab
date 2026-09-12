"""One isolated simulation, including interventions and its own neural state."""
from __future__ import annotations

import math
import numpy as np
import torch
from .neural import ReferenceBrain, REGIONS, N
from .body import FlyBody
from .arena import odor_sensors, bearing_error


class Simulation:
    def __init__(self):
        self.brain = ReferenceBrain()
        self.body = FlyBody()
        self.running = False
        self.speed = 1
        self.odor = "A"
        self.intensity = .7
        self.selected = "mushroom"
        self.environment_mode = "uniform"
        self.source = np.array([12., 3., .01])
        self.silenced: set[str] = set()
        self.pulses: dict[str, tuple[float, float]] = {}
        self.reset()

    def reset(self):
        self.body.reset()
        self.state = torch.zeros(N)
        self.time = 0.
        self.steps = 0
        self.history = []
        self.pulses.clear()
        self.silenced.clear()

    def stimulate(self, region, amplitude=1., duration=.5):
        self.pulses[region] = (amplitude, self.time + duration)

    @torch.no_grad()
    def advance(self, count=1):
        for _ in range(count):
            # Synthetic sensory encoder; cue is explicit for conditioning trials.
            concentration = float(odor_sensors(self.body, self.source, self.intensity).mean()) if self.environment_mode == "spatial" else self.intensity
            obs = torch.tensor([.25, concentration if self.odor == "A" else 0., concentration if self.odor == "B" else 0., float(np.clip(self.body.foot_feedback().mean() / 3, 0, 1))])
            stim, mask = torch.zeros(N), torch.ones(N)
            for r in REGIONS:
                a, b = r["start"], r["end"]
                pulse = self.pulses.get(r["id"])
                if pulse and pulse[1] > self.time:
                    stim[a:b] = pulse[0]
                if r["id"] in self.silenced:
                    mask[a:b] = 0
            # Reference VNC receives an explicit oscillator: not claimed to emerge
            # from the measured connectome. All muscle commands pass through neurons.
            contacts = np.clip(self.body.foot_feedback() / 5, 0, 1)
            for leg in range(6):
                phase = self.body.phases[leg]
                stim[72+leg*2:74+leg*2] += torch.tensor([math.sin(phase), -math.sin(phase)]) * .7
                stim[72 + leg * 2] += float(contacts[leg]) * .12
            self.state = self.brain.step(self.state, obs, stim, mask)
            drive = self.state[72:84] + self.state[84:96] * .3
            # Twelve motor-group drives modulate six leg synergies. The body controller
            # expands these into 42 antagonistic pairs with joint feedback.
            dn = self.brain.logits(self.state).softmax(-1)[0]
            motor = torch.sigmoid(3*drive) * (.25 + .75*dn)
            if self.environment_mode == "spatial" and self.odor != "none":
                # Explicit steering decoder: learned approach/avoid preference sets
                # the desired bearing; differential leg drive turns the physical body.
                preference = 2 * float(dn) - 1
                error = bearing_error(self.body, self.source)
                if preference < 0:
                    error = math.atan2(math.sin(error + math.pi), math.cos(error + math.pi))
                turn = np.clip(error, -1, 1) * abs(preference)
                motor[:6] *= float(1 - .4 * turn)
                motor[6:] *= float(1 + .4 * turn)
            motor *= mask[84:96]
            # Silencing VNC removes its locomotor drive, leaving the body passive.
            if "vnc" in self.silenced:
                motor.zero_()
            self.state[84:96] = motor
            self.body.step(motor.numpy())
            self.time += .02
            self.steps += 1
            self.history.append({"time": self.time, "neural": float(self.state[32:64].abs().mean()), "muscle": float(np.mean(self.body.data.act)), "speed": float(np.linalg.norm(self.body.data.qvel[:2]))})
            self.history = self.history[-300:]

    def snapshot(self):
        return {
            "time": self.time, "steps": self.steps, "running": self.running, "speed": self.speed,
            "odor": self.odor, "intensity": self.intensity, "selected": self.selected,
            "silenced": sorted(self.silenced),
            "regions": [{**r, "activity": float(self.state[r["start"]:r["end"]].abs().mean()), "stimulated": r["id"] in self.pulses and self.pulses[r["id"]][1] > self.time} for r in REGIONS],
            "neurons": self.state.tolist(), "body": self.body.snapshot(), "history": self.history,
            "environment": {"mode": self.environment_mode, "source": self.source.tolist(),
                            "distance": float(np.linalg.norm(self.body.data.qpos[:2] - self.source[:2])),
                            "concentration": (float(odor_sensors(self.body, self.source, self.intensity).mean()) if self.environment_mode == "spatial" else self.intensity) if self.odor != "none" else 0.,
                            "antennae": (odor_sensors(self.body, self.source, self.intensity).tolist() if self.environment_mode == "spatial" else [self.intensity] * 2) if self.odor != "none" else [0., 0.]},
            "approach_probability": float(self.brain.logits(self.state).softmax(-1)[0]),
            "model": {"name": "Synthetic reference circuit", "neurons": N, "edges": self.brain.edge_count, "engine": "PyTorch + MuJoCo", "measured_connectome": False},
        }
