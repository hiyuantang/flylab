"""Experimental GPU matrix readout of the existing peripheral mapping.

This is not a replacement neural network: rows encode the bridge's existing
weighted channel means. No motor neuron or transmission is selected away.
Not enabled in live checkpoints; validate and benchmark before integration.
"""
import numpy as np
import torch
from .motor_mapping import LEGACY_PROFILE


class MatrixMotorReadout:
    def __init__(self, bridge, device='mps'):
        if bridge.mapping_profile == LEGACY_PROFILE:
            raise ValueError('Legacy unweighted half means require their original rounding')
        self.bridge = bridge
        ids = sorted({int(i) for indices in bridge.channels for i in indices})
        lookup = {i: j for j, i in enumerate(ids)}
        matrix = torch.zeros((len(bridge.channels), len(ids)), dtype=torch.float32)
        for channel, (indices, weights) in enumerate(zip(bridge.channels, bridge.channel_weights)):
            for index, weight in zip(indices.tolist(), weights.tolist()):
                matrix[channel, lookup[index]] += weight / len(indices)
        self.ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.matrix = matrix.to(device)

    def __call__(self):
        # Read only the mapped motor rates on GPU, then one dense matrix-vector
        # product returns all actuator means. Only the small output crosses back.
        rates = self.bridge.brain.rates.detach()[self.ids].float()
        if not len(self.ids):
            return np.zeros(len(self.bridge.channels), dtype=np.float32)
        means = (self.matrix @ rates).cpu().double().numpy()
        # Keep the baseline's CPU double gain/clamp and final float32 conversion.
        return np.clip(means / 100 * np.exp(self.bridge.parameters[7]), 0, 1).astype(np.float32)
