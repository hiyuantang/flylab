"""Synthetic reference circuit. Never presented as measured MaleCNS connectivity.

Rate dynamics are used here; a connectome-backed LIF model is in connectome.py.
Only MB -> descending synaptic efficacy is trainable in this reference circuit.
"""
from __future__ import annotations

import torch
from torch import nn

REGIONS = [
    {"id": "optic", "name": "Optic lobes", "start": 0, "end": 16},
    {"id": "antennal", "name": "Antennal lobes", "start": 16, "end": 32},
    {"id": "mushroom", "name": "Mushroom body", "start": 32, "end": 64},
    {"id": "descending", "name": "Descending", "start": 64, "end": 72},
    {"id": "vnc", "name": "Ventral nerve cord", "start": 72, "end": 84},
    {"id": "motor", "name": "Motor neurons", "start": 84, "end": 96},
]
N = 96


class ReferenceBrain(nn.Module):
    def __init__(self, seed: int = 7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        edges = torch.zeros(N, N)
        for a, b in [(0, 16), (16, 32), (32, 64), (64, 72), (72, 84)]:
            end_a = next(r["end"] for r in REGIONS if r["start"] == a)
            end_b = next(r["end"] for r in REGIONS if r["start"] == b)
            block = (torch.rand(end_b-b, end_a-a, generator=g) < .4).float()
            edges[b:end_b, a:end_a] = block * .10
        # Local feedback plus a descending-to-MB recurrent pathway.
        edges += torch.eye(N) * .12
        edges[32:64, 64:72] = .015
        edges[64:72, 32:64] = 0  # separately represented plastic connections
        self.register_buffer("fixed", edges)
        sensory = torch.zeros(N, 4)
        sensory[:16, 0] = .2
        sensory[16:24, 1] = 1.5
        sensory[24:32, 2] = 1.5
        # Distinct odor-responsive populations for a deliberately small learning task.
        sensory[32:48, 1] = 1.5
        sensory[48:64, 2] = 1.5
        sensory[72:84, 3] = .2
        self.register_buffer("sensory", sensory)
        self.plastic = nn.Parameter(torch.randn(8, 32, generator=g) * .025)

    def step(self, state, observation, stimulation=None, silenced=None):
        drive = state @ self.fixed.T + observation @ self.sensory.T
        plastic_drive = state[..., 32:64] @ self.plastic.T
        # Smooth signed rate units are a modeling assumption, not a neurotransmitter map.
        drive = drive + torch.nn.functional.pad(plastic_drive, (64, 24))
        if stimulation is not None:
            drive = drive + stimulation
        new = state + .25 * (torch.tanh(drive) - state)
        if silenced is not None:
            new = new * silenced
        return new

    def settle(self, observation, steps=12):
        state = torch.zeros((*observation.shape[:-1], N), device=observation.device)
        for _ in range(steps):
            state = self.step(state, observation)
        return state

    @staticmethod
    def logits(state):
        return torch.stack((state[..., 64:68].mean(-1), state[..., 68:72].mean(-1)), -1) * 6

    @property
    def edge_count(self):
        return int(torch.count_nonzero(self.fixed)) + self.plastic.numel()
