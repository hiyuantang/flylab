"""One isolated simulation, including interventions and its own neural state."""
from __future__ import annotations

import math
import time
from dataclasses import replace
import numpy as np
import torch
from .neural import ReferenceBrain, REGIONS, N
from .body import FlyBody
from .arena import odor_sensors, bearing_error
from pathlib import Path
from .full_connectome import FullBrain, Physiology
from .neuromuscular import NeuromuscularBridge
from .motor_mapping import MAPPING_PROFILE, LEGACY_PROFILE, validate_body_profile
from .body import BodyParameters
from .physical_policy import validate_policy
from .senses import SensorSuite, SensorySettings
from .scenes import get_scene, scene_summary


class Simulation:
    def __init__(self):
        self.brain = ReferenceBrain()
        self.body = FlyBody()
        self.running = False
        self.speed = 1
        self.episode = 0
        self.odor = "A"
        self.intensity = .7
        self.selected = "mushroom"
        self.environment_mode = "uniform"
        self.source = np.array([12., 3., .01])
        self.controller = 'synthetic'
        self.full_brain = None
        self.bridge = None
        self.posture_gains = np.zeros(8, dtype=np.float32)
        self.sensors = SensorSuite(SensorySettings(vision_enabled=True))
        self.silenced: set[str] = set()
        self.pulses: dict[str, tuple[float, float]] = {}
        self.reset()

    def reset(self):
        self.episode += 1
        self.body.reset()
        self.pose_target = self.body.data.qpos[self.body.qadr].copy()
        if self.full_brain is not None:
            self.full_brain.reset()
            self.bridge.set_parameters(self.bridge.parameters)
        self.state = torch.zeros(N)
        self.time = 0.
        self.steps = 0
        self.history = []
        self.wall_seconds = 0.
        self.last_step_wall_seconds = 0.
        self.pulses.clear()
        self.silenced.clear()

    def sensory_frame(self):
        return self.sensors.sample(self.body, self.source, self.intensity if self.odor != 'none' else 0., self.environment_mode == 'spatial')

    def configure_senses(self, settings):
        self.sensors = SensorSuite(settings)
        self.running = False
        self.reset()

    def configure_scene(self, scene_id):
        scene = get_scene(scene_id)
        body = FlyBody(self.body.parameters, scene_id)
        settings = replace(self.sensors.settings, stimulus_position=tuple(scene['sound_source']))
        self.running = False
        self.body, self.sensors = body, SensorSuite(settings)
        self.source = np.array(scene['odor_source'], dtype=float)
        self.environment_mode = 'uniform' if scene_id == 'lab' else 'spatial'
        self.reset()

    def configure(self, controller, directory=None, physiology=None, body_parameters=None, scene_id=None, device=None, mapping_profile=None):
        if controller not in {'synthetic', 'posture', 'connectome'}:
            raise ValueError('Unknown controller')
        # Prepare replacements before mutating the running workbench.
        body = FlyBody(body_parameters or self.body.parameters, scene_id or self.body.scene_id)
        brain = self.full_brain
        bridge = self.bridge
        mapping_profile = mapping_profile or (bridge.mapping_profile if bridge else MAPPING_PROFILE)
        if controller == 'connectome':
            validate_body_profile(mapping_profile, body.parameters)
        directory = Path(directory or (brain.directory if brain else 'data/full')).resolve()
        physiology = physiology or (brain.config if brain else Physiology.paper())
        device = device or (brain.voltage.device.type if brain else 'cpu')
        if controller == 'connectome' and not math.isclose(round(20 / physiology.dt_ms) * physiology.dt_ms, 20):
            raise ValueError('Neural integration time must divide the 20 ms body step')
        if controller == 'connectome' and (brain is None or brain.config != physiology or brain.directory.resolve() != directory or brain.voltage.device.type != device):
            brain = FullBrain(directory, physiology, device=device)
            bridge = NeuromuscularBridge(brain, mapping_profile)
            if self.bridge is not None and brain.manifest['graph_sha256'] == self.full_brain.manifest['graph_sha256']:
                bridge.set_parameters(self.bridge.parameters)
        elif controller == 'connectome' and bridge.mapping_profile != mapping_profile:
            bridge = NeuromuscularBridge(brain, mapping_profile)
            bridge.set_parameters(self.bridge.parameters)
        self.running = False
        self.body, self.full_brain, self.bridge = body, brain, bridge
        self.controller = controller
        self.reset()

    def load_physical(self, checkpoint, directory):
        values, body, physiology, environment = validate_policy(checkpoint)
        if checkpoint['mode'] == 'connectome':
            import json
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest['graph_sha256'] != checkpoint['graph_sha256']:
                raise ValueError('Checkpoint belongs to a different anatomical graph')
        self.configure(checkpoint['mode'], directory, physiology, body, environment.scene_id,
                       mapping_profile=checkpoint.get('mapping_profile', LEGACY_PROFILE))
        if self.controller == 'posture':
            self.posture_gains = values.copy()
        else:
            self.bridge.set_parameters(values)
        self.environment_mode, self.odor, self.intensity = environment.mode, environment.odor, environment.intensity
        self.source = np.array(environment.source, dtype=float)
        self.sensors = SensorSuite(environment.senses)

    def stimulate(self, region, amplitude=1., duration=.5):
        self.pulses[region] = (amplitude, self.time + duration)

    @torch.no_grad()
    def advance(self, count=1):
        for _ in range(count):
            started = time.perf_counter()
            if self.controller != 'synthetic':
                self._advance_physical()
                self.last_step_wall_seconds = time.perf_counter() - started
                self.wall_seconds += self.last_step_wall_seconds
                continue
            # Synthetic sensory encoder; cue is explicit for conditioning trials.
            concentration = float(odor_sensors(self.body, self.source, self.intensity).mean()) if self.environment_mode == "spatial" else self.intensity
            frame = self.sensory_frame()
            obs = torch.tensor([0., concentration if self.odor == "A" else 0., concentration if self.odor == "B" else 0., float(np.mean(frame['touch']))])
            stim, mask = torch.zeros(N), torch.ones(N)
            # Synthetic hemispheres accept coarse vision and antennal mechanics.
            # This encoding is distinct from the annotation-backed MaleCNS inputs.
            for side in range(2):
                stim[side * 8:(side + 1) * 8] += float(frame['vision']['mean'][side])
                stim[64 + side * 4:68 + side * 4] += float(frame['hearing'][side] + frame['wind'][side]) * .3
            for r in REGIONS:
                a, b = r["start"], r["end"]
                pulse = self.pulses.get(r["id"])
                if pulse and pulse[1] > self.time:
                    stim[a:b] += pulse[0]
                if r["id"] in self.silenced:
                    mask[a:b] = 0
            # Reference VNC receives an explicit oscillator: not claimed to emerge
            # from the measured connectome. All muscle commands pass through neurons.
            contacts = frame['touch']
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
            self.last_step_wall_seconds = time.perf_counter() - started
            self.wall_seconds += self.last_step_wall_seconds

    def _advance_physical(self):
        if self.controller == 'posture':
            self.body.step_posture(self.pose_target, self.posture_gains)
            self.state.zero_()
        else:
            pulses = {k: v[0] for k, v in self.pulses.items() if v[1] > self.time}
            self.bridge.step(self.body, self.source, self.intensity if self.odor != 'none' else 0.,
                             self.environment_mode == 'spatial', pulses, self.silenced, frame=self.sensory_frame())
            rates = self.full_brain.rates.detach().cpu()
            for region in REGIONS:
                indices = self.bridge.regions[region['id']]
                rate = float(rates[indices].mean()) / 100 if len(indices) else 0.
                self.state[region['start']:region['end']] = min(1., rate)
        self.time += .02
        self.steps += 1
        self.history.append({'time': self.time, 'neural': float(self.state[32:64].abs().mean()),
                             'muscle': float(np.mean(self.body.data.act)), 'speed': float(np.linalg.norm(self.body.data.qvel[:2]))})
        self.history = self.history[-300:]

    def snapshot(self):
        return {
            "time": self.time, "steps": self.steps, "episode": self.episode, "running": self.running, "speed": self.speed,
            'timing': {'simulated_seconds': self.time, 'compute_wall_seconds': self.wall_seconds,
                       'last_step_wall_seconds': self.last_step_wall_seconds,
                       'simulated_per_wall_second': self.time / self.wall_seconds if self.wall_seconds else None,
                       'neural_dt_ms': self.full_brain.config.dt_ms if self.controller == 'connectome' else None,
                       'policy': 'Every integration step is computed. Requested rate is a pacing ceiling; slow hardware never drops neural or physical steps.'},
            "senses": {**self.sensory_frame(), 'retinal_routing': self.bridge.retina().summary if self.bridge and self.sensors.settings.vision_model == 'compound-retina-v1' else None, 'neural_routing': self.bridge.summary()['sensory_neurons'] if self.controller == 'connectome' else None},
            "scene": scene_summary(self.body.scene_id),
            "odor": self.odor, "intensity": self.intensity, "selected": self.selected,
            "silenced": sorted(self.silenced),
            "regions": [{**r, 'count': len(self.bridge.regions[r['id']]) if self.controller == 'connectome' else r['end'] - r['start'], "activity": float(self.state[r["start"]:r["end"]].abs().mean()), "stimulated": r["id"] in self.pulses and self.pulses[r["id"]][1] > self.time} for r in REGIONS],
            "neurons": self.state.tolist() if self.controller == 'synthetic' else [], "body": self.body.snapshot(), "history": self.history,
            "environment": {"mode": self.environment_mode, "source": self.source.tolist(),
                            "distance": float(np.linalg.norm(self.body.data.qpos[:2] - self.source[:2])),
                            "concentration": (float(odor_sensors(self.body, self.source, self.intensity).mean()) if self.environment_mode == "spatial" else self.intensity) if self.odor != "none" else 0.,
                            "antennae": (odor_sensors(self.body, self.source, self.intensity).tolist() if self.environment_mode == "spatial" else [self.intensity] * 2) if self.odor != "none" else [0., 0.]},
            "approach_probability": float(self.brain.logits(self.state).softmax(-1)[0]) if self.controller == 'synthetic' else None,
            "model": {"name": {'synthetic': 'Synthetic reference circuit', 'posture': 'Posture feedback baseline', 'connectome': 'MaleCNS annotated full graph'}[self.controller],
                      'controller': self.controller, "neurons": len(self.full_brain.ids) if self.controller == 'connectome' else (N if self.controller == 'synthetic' else 0),
                      "edges": self.full_brain.manifest['edges'] if self.controller == 'connectome' else (self.brain.edge_count if self.controller == 'synthetic' else 0),
                      "engine": "PyTorch + MuJoCo", "measured_connectome": self.controller == 'connectome',
                      'device': self.full_brain.voltage.device.type if self.controller == 'connectome' else 'cpu', 'precision': str(self.full_brain.voltage.dtype) if self.controller == 'connectome' else 'torch.float32',
                      'execution': self.full_brain.execution_summary() if self.controller == 'connectome' and self.full_brain.config.profile == 'shiu-2024' else None,
                      'dynamics_version': self.full_brain.dynamics_version if self.controller == 'connectome' else None,
                      'neural_wall_seconds': self.full_brain.last_wall_seconds if self.controller == 'connectome' else 0.,
                      'spikes': self.full_brain.total_spikes if self.controller == 'connectome' else 0,
                      'assumptions': 'Measured wiring; assumed physiology and partial muscle routing' if self.controller == 'connectome' else 'Engineering baseline'},
        }
