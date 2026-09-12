"""One isolated simulation, including interventions and its own neural state."""
from __future__ import annotations

import math
import numpy as np
import torch
from .neural import ReferenceBrain, REGIONS, N
from .body import FlyBody


class Simulation:
    def __init__(self):
        self.brain = ReferenceBrain()
        self.body = FlyBody()
        self.running = False
        self.speed = 1
        self.odor = "A"
        self.intensity = .7
        self.selected = "mushroom"
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
            obs = torch.tensor([.25, self.intensity if self.odor == "A" else 0., self.intensity if self.odor == "B" else 0., self.body.snapshot()["speed"]])
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
            for leg in range(6):
                phase = self.time * 2*math.pi*3 + (0 if leg in [0, 2, 4] else math.pi)
                stim[72+leg*2:74+leg*2] += torch.tensor([math.sin(phase), -math.sin(phase)]) * .7
            self.state = self.brain.step(self.state, obs, stim, mask)
            drive = self.state[72:84] + self.state[84:96] * .3
            # Six antagonistic pairs: VNC drive becomes motor-population activity.
            dn = self.brain.logits(self.state).softmax(-1)[0]
            motor = torch.sigmoid(3*drive) * (.25 + .75*dn)
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
            "approach_probability": float(self.brain.logits(self.state).softmax(-1)[0]),
            "model": {"name": "Synthetic reference circuit", "neurons": N, "edges": self.brain.edge_count, "engine": "PyTorch + MuJoCo", "measured_connectome": False},
        }
