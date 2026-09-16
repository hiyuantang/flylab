"""One isolated simulation, including interventions and its own neural state."""
from __future__ import annotations

import math
import time
from dataclasses import replace
import numpy as np
import torch
from .neural import ReferenceBrain, REGIONS, N
from .coupling import run_pair, COUPLING_VERSION
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
        self.command_hz = 50
        self.coupling_mode = 'serial'
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
            self.bridge.locomotion.reset()
            self.bridge.set_parameters(self.bridge.parameters)
            if getattr(self, '_gesture_adapter', None) is not None:
                self._gesture_adapter.apply(self.gesture_adapter['parameters'].numpy())
        self.state = torch.zeros(N)
        self.time = 0.
        self.steps = 0
        self.clock_origin = 0.
        self.clock_step_origin = 0
        self.history = []
        self.wall_seconds = 0.
        self.last_step_wall_seconds = 0.
        self._reset_command_buffer()
        self.pulses.clear()
        self.silenced.clear()

    def _reset_command_buffer(self):
        self.pending_command = self.body.data.ctrl.astype(np.float32).copy()
        self.command_generated_at = self.time
        self.command_source_time = None
        self.applied_command_generated_at = None
        self.last_sensory_time = None
        self.cycle_incomplete = False

    def set_coupling(self, mode):
        if mode not in {'serial', 'pipelined'}:
            raise ValueError('Coupling modes: serial, pipelined')
        if self.cycle_incomplete:
            raise ValueError('Restore or reset the incomplete cycle before changing execution')
        if mode == 'pipelined' and (self.controller != 'connectome' or self.full_brain.config.profile != 'shiu-2024'):
            raise ValueError('Pipelined coupling requires the measured paper-profile brain')
        if mode != self.coupling_mode:
            self.coupling_mode = mode
            self._reset_command_buffer()

    def coupling_summary(self):
        pipelined = self.coupling_mode == 'pipelined'
        return {'coupling_mode': self.coupling_mode,
                'command_delay_ms': 1000 / self.command_hz if pipelined else 0.,
                'command_generated_at': self.command_generated_at if pipelined else None,
                'command_source_time': self.command_source_time if pipelined else None,
                'applied_command_generated_at': self.applied_command_generated_at if pipelined else None,
                'last_sensory_time': self.last_sensory_time if pipelined else None}

    def coupling_state(self):
        value = {'version': COUPLING_VERSION, 'mode': self.coupling_mode}
        if self.coupling_mode == 'pipelined':
            if (self.pending_command.shape != (self.body.model.nu,)
                    or self.pending_command.dtype != np.float32
                    or not np.isfinite(self.pending_command).all()
                    or np.any((self.pending_command < 0) | (self.pending_command > 1))
                    or not math.isclose(self.command_generated_at, self.time, abs_tol=1e-9)
                    or any(t is not None and (not math.isfinite(t) or not 0 <= t <= self.time)
                           for t in (self.command_source_time, self.applied_command_generated_at, self.last_sensory_time))):
                raise ValueError('Invalid buffered muscle command or timestamp')
            value.update(command=torch.from_numpy(self.pending_command.copy()),
                         generated_at=self.command_generated_at, source_time=self.command_source_time,
                         applied_at=self.applied_command_generated_at, sensed_at=self.last_sensory_time)
        return value

    def sensory_frame(self):
        return self.sensors.sample(self.body, self.source, self.intensity if self.odor != 'none' else 0., self.environment_mode == 'spatial', duration_s=1 / self.command_hz)

    def set_command_hz(self, hz, neural_dt_ms=None):
        if hz not in {50, 60} or self.controller != 'connectome':
            raise ValueError('Measured workbench command rates: 50 or 60 Hz')
        dt = neural_dt_ms if neural_dt_ms is not None else (1000 / 960 if hz == 60 else 1.)
        interval = 1000 / hz
        if not math.isclose(round(interval / dt) * dt, interval):
            raise ValueError('Neural timestep must divide the muscle-command interval')
        self.full_brain.set_timestep(dt)
        self.command_hz, self.clock_origin, self.clock_step_origin = hz, self.time, self.steps

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

    def prepare_standing_trial(self):
        """Explicit elastic-support baseline with the full neural controller.

        This is a new experiment, not a continuation or a learned standing
        behavior. Springs model assumed passive joint elasticity; there is no
        runtime pose target, direct motor stimulus or external support force.
        """
        if self.controller != 'connectome' or self.full_brain.config.profile != 'shiu-2024':
            raise ValueError('Standing trial requires the full measured brain')
        body = FlyBody(replace(self.body.parameters, elasticity_profile='stance-elastic-v1',
                              mass_scale=1., strength_scale=1., friction=1.), 'lab')
        sensors = SensorSuite(replace(self.sensors.settings,
            proprioception_model='feco-opponent-v1', contact_model='tarsal-contact-v1', spatial_model='geometry-v3',
            stimulus_position=tuple(body.scene['sound_source']),
            vision_enabled=False, hearing_enabled=False, wind_enabled=False,
            proprioception_enabled=True, touch_enabled=True))
        self.running = False
        self.body, self.sensors = body, sensors
        self.source = np.array(body.scene['odor_source'], dtype=float)
        self.environment_mode, self.odor = 'uniform', 'none'
        parameters = np.zeros(8, dtype=np.float32)
        parameters[7] = -2.  # Explicit rate-to-muscle calibration, exp(-2)=0.135.
        self.bridge.set_parameters(parameters)
        self.reset()

    def configure(self, controller, directory=None, physiology=None, body_parameters=None, scene_id=None, device=None, mapping_profile=None, command_hz=None, precision=None, coupling_mode=None):
        if controller not in {'synthetic', 'posture', 'connectome'}:
            raise ValueError('Unknown controller')
        if coupling_mode is not None and coupling_mode not in {'serial', 'pipelined'}:
            raise ValueError('Coupling modes: serial, pipelined')
        # Prepare replacements before mutating the running workbench.
        body = FlyBody(body_parameters or self.body.parameters, scene_id or self.body.scene_id)
        brain = self.full_brain
        bridge = self.bridge
        mapping_profile = mapping_profile or (bridge.mapping_profile if bridge else MAPPING_PROFILE)
        if controller == 'connectome':
            validate_body_profile(mapping_profile, body.parameters)
        directory = Path(directory or (brain.directory if brain else 'data/full')).resolve()
        physiology = physiology or (brain.config if brain else Physiology.paper())
        if coupling_mode == 'pipelined' and (controller != 'connectome' or physiology.profile != 'shiu-2024'):
            raise ValueError('Pipelined coupling requires the measured paper-profile brain')
        device = device or (brain.voltage.device.type if brain else 'cpu')
        if physiology.profile != 'shiu-2024':
            precision = precision or 'float32'
        precision = precision or (str(brain.voltage.dtype).removeprefix('torch.') if brain and brain.voltage.device.type == device
                                  else 'float32' if device == 'mps' else 'float64')
        hz = command_hz or self.command_hz
        if hz not in {50, 60}:
            raise ValueError('Supported command rates: 50 or 60 Hz')
        interval = 1000 / hz
        if controller == 'connectome' and not math.isclose(round(interval / physiology.dt_ms) * physiology.dt_ms, interval):
            raise ValueError('Neural integration time must divide the muscle-command interval')
        if controller == 'connectome' and (brain is None or brain.config != physiology or brain.directory.resolve() != directory or brain.voltage.device.type != device or str(brain.voltage.dtype) != f'torch.{precision}'):
            brain = FullBrain(directory, physiology, device=device, precision=precision)
            bridge = NeuromuscularBridge(brain, mapping_profile)
            if self.bridge is not None and brain.manifest['graph_sha256'] == self.full_brain.manifest['graph_sha256']:
                bridge.set_parameters(self.bridge.parameters)
        elif controller == 'connectome' and bridge.mapping_profile != mapping_profile:
            bridge = NeuromuscularBridge(brain, mapping_profile)
            bridge.set_parameters(self.bridge.parameters)
        if getattr(self, '_gesture_adapter', None) is not None:
            self._gesture_adapter.apply(self._gesture_adapter.initial)
        self._gesture_adapter = None
        self.gesture_adapter = None
        self.gesture_model = None
        self.running = False
        self.body, self.full_brain, self.bridge = body, brain, bridge
        self.controller = controller
        self.command_hz = hz
        self.coupling_mode = coupling_mode or (self.coupling_mode if controller == 'connectome' and physiology.profile == 'shiu-2024' else 'serial')
        self.reset()

    def load_physical(self, checkpoint, directory):
        values, body, physiology, environment = validate_policy(checkpoint)
        if checkpoint['mode'] == 'connectome':
            import json
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest['graph_sha256'] != checkpoint['graph_sha256']:
                raise ValueError('Checkpoint belongs to a different anatomical graph')
        self.configure(checkpoint['mode'], directory, physiology, body, environment.scene_id,
                       mapping_profile=checkpoint.get('mapping_profile', LEGACY_PROFILE), command_hz=50, coupling_mode='serial')
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
            frame = self.sensory_frame()
            concentration = float(np.mean(frame['odor']))
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
            if self.cycle_incomplete:
                raise RuntimeError('Previous cycle is incomplete; restore or reset before advancing')
            frame = self.sensory_frame()
            reward = self.full_brain.reward_circuit if self.full_brain.config.profile == 'shiu-2024' else None
            poisson = reward.input_rates(1 / self.command_hz) if reward else None
            if self.coupling_mode == 'pipelined':
                # Freeze sensory input before any MuJoCo mutation. Neither brain
                # preparation nor computation reads a concurrently moving body.
                drive, mask = self.bridge.prepare_command(self.body, self.source,
                    self.intensity if self.odor != 'none' else 0., self.environment_mode == 'spatial',
                    pulses, self.silenced, frame=frame)
                held = self.pending_command.copy()
                generated_at = self.command_generated_at
                duration = 1 / self.command_hz
                self.cycle_incomplete = True
                command = run_pair(
                    lambda: self.bridge.advance_command(drive, mask, self.silenced, duration, poisson_hz=poisson),
                    lambda: self.body.step_muscles(held, dt=duration))
                command = np.asarray(command, dtype=np.float32)
                if command.shape != (self.body.model.nu,) or not np.isfinite(command).all() or np.any((command < 0) | (command > 1)):
                    raise RuntimeError('Invalid buffered muscle command')
                self.pending_command = command.copy()
                self.command_generated_at = self.clock_origin + (self.steps + 1 - self.clock_step_origin) / self.command_hz
                self.command_source_time = self.last_sensory_time = float(frame['time'])
                self.applied_command_generated_at = generated_at
                self.cycle_incomplete = False
            elif poisson is not None:
                drive, mask = self.bridge.prepare_command(self.body, self.source,
                    self.intensity if self.odor != 'none' else 0., self.environment_mode == 'spatial',
                    pulses, self.silenced, frame=frame)
                self.cycle_incomplete = True
                action = self.bridge.advance_command(drive, mask, self.silenced, 1 / self.command_hz, poisson_hz=poisson)
                self.body.step_muscles(action, dt=1 / self.command_hz)
                self.cycle_incomplete = False
            else:
                self.bridge.step(self.body, self.source, self.intensity if self.odor != 'none' else 0.,
                                 self.environment_mode == 'spatial', pulses, self.silenced,
                                 frame=frame, duration_s=1 / self.command_hz)
            rates = self.full_brain.rates.detach().cpu()
            if reward:
                reward.update(rates, 1 / self.command_hz)
            for region in REGIONS:
                indices = self.bridge.regions[region['id']]
                rate = float(rates[indices].mean()) / 100 if len(indices) else 0.
                self.state[region['start']:region['end']] = min(1., rate)
        self.steps += 1
        self.time = self.clock_origin + (self.steps - self.clock_step_origin) / self.command_hz
        self.history.append({'time': self.time, 'neural': float(self.state[32:64].abs().mean()),
                             'muscle': float(np.mean(self.body.data.act)), 'speed': float(np.linalg.norm(self.body.data.qvel[:2]))})
        self.history = self.history[-300:]

    def snapshot(self):
        reward = self.full_brain.reward_circuit if self.controller == "connectome" and self.full_brain.config.profile == "shiu-2024" else None
        senses = self.sensory_frame()
        routing = self.bridge.summary()['sensory_neurons'] if self.controller == 'connectome' else None
        leg_routing = self.bridge.leg_feedback.summary() if self.bridge and senses.get('legs') else None
        if routing and leg_routing:
            routing['proprioceptive'] = leg_routing['neurons']
        senses.update(neural_routing=routing, leg_routing=leg_routing,
                      retinal_routing=self.bridge.retina().summary if self.bridge and self.sensors.settings.vision_model == 'compound-retina-v1' else None)
        return {
            'gesture_stimulus': self.sensors.visual_object.summary() if self.sensors.visual_object else None,
            'gesture_model': getattr(self, 'gesture_model', None),
            'locomotion': self.bridge.locomotion.summary(self.time) if self.controller == 'connectome' else None,
            "reward": reward.summary() if reward else None,
            "time": self.time, "steps": self.steps, "episode": self.episode, "running": self.running, "speed": self.speed,
            'timing': {'simulated_seconds': self.time, 'compute_wall_seconds': self.wall_seconds,
                       'last_step_wall_seconds': self.last_step_wall_seconds,
                       'simulated_per_wall_second': self.time / self.wall_seconds if self.wall_seconds else None,
                       'neural_dt_ms': self.full_brain.config.dt_ms if self.controller == 'connectome' else None,
                       'muscle_command_hz': self.command_hz,
                       'sensory_hz': self.command_hz,
                       'coupling': self.coupling_summary(),
                       'physics_max_dt_ms': self.body.model.opt.timestep * 1000,
                       'policy': 'Every integration step is computed. Requested rate is a pacing ceiling; slow hardware never drops neural or physical steps.'},
            'senses': senses,
            "scene": scene_summary(self.body.scene_id),
            "odor": self.odor, "intensity": self.intensity, "selected": self.selected,
            "silenced": sorted(self.silenced),
            "regions": [{**r, 'count': len(self.bridge.regions[r['id']]) if self.controller == 'connectome' else r['end'] - r['start'], "activity": float(self.state[r["start"]:r["end"]].abs().mean()), "stimulated": r["id"] in self.pulses and self.pulses[r["id"]][1] > self.time} for r in REGIONS],
            "neurons": self.state.tolist() if self.controller == 'synthetic' else [], "body": self.body.snapshot(), "history": self.history,
            "environment": {"mode": self.environment_mode, "source": self.source.tolist(),
                            "distance": float(np.linalg.norm(self.body.data.qpos[:(3 if self.sensors.settings.spatial_model == 'geometry-v3' else 2)] - self.source[:(3 if self.sensors.settings.spatial_model == 'geometry-v3' else 2)])),
                            "concentration": float(np.mean(senses['odor'])),
                            "antennae": senses['odor']},
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
