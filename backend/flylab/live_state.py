"""Atomic, complete continuation checkpoints for the measured workbench."""
from dataclasses import asdict
import hashlib
import math
import pickle
from pathlib import Path
import tempfile

import mujoco
import numpy as np
import torch

from .body import BodyParameters
from .coupling import COUPLING_VERSION
from .full_connectome import Physiology
from .paper_dynamics import PAPER_DYNAMICS_VERSION
from .metal_dynamics import METAL_VERSION, ACTIVE_HALF_METAL_VERSION, NEURAL_TENSORS
from .motor_mapping import LEGACY_PROFILE
from .physical_policy import TrialEnvironment
from .senses import SENSORY_VERSION, SensorSuite

BODY_TENSORS = ('phases', 'magnitudes', 'command', 'target')
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def model_fingerprint(body):
    # Hash the compiled model, including meshes, parameters and solver options.
    buffer = np.empty(mujoco.mj_sizeModel(body.model), dtype=np.uint8)
    mujoco.mj_saveModel(body.model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def save_live(sim, path: Path):
    if sim.controller != 'connectome' or sim.full_brain.config.profile != 'shiu-2024':
        raise ValueError('Live continuation requires the full measured paper-profile brain')
    brain = sim.full_brain
    if (sim.cycle_incomplete or not math.isclose(brain.time_ms, sim.time * 1000., abs_tol=1e-8)
            or not math.isclose(sim.body.data.time, sim.time, abs_tol=1e-9)
            or any(not torch.isfinite(getattr(brain, name)).all() for name in NEURAL_TENSORS)):
        raise ValueError('Refusing to overwrite a saved life state with an incomplete or invalid step')
    state = np.empty(mujoco.mj_stateSize(sim.body.model, STATE_SPEC))
    mujoco.mj_getState(sim.body.model, sim.body.data, state, STATE_SPEC)
    if not np.isfinite(state).all():
        raise ValueError('Cannot save non-finite physical state')
    reward = brain.__dict__.get('reward_circuit')
    reward_state = reward.state_dict() if reward and reward.available else None
    walking = sim.bridge.locomotion.state_dict()
    walking_active = walking['amplitude'] > 0 and walking['expires'] > sim.time
    payload = {
        'locomotion': walking,
        'dopamine': reward_state,
        'format': 'flylab-live-v4' if walking_active else 'flylab-live-v3' if reward_state is not None else 'flylab-live-v2' if sim.coupling_mode == 'pipelined' else 'flylab-live-v1',
        'coupling': sim.coupling_state(), 'dynamics_version': brain.dynamics_version,
        'sensory_version': SENSORY_VERSION, 'mujoco_version': mujoco.__version__,
        'body_sha256': model_fingerprint(sim.body),
        'graph_sha256': brain.manifest['graph_sha256'], 'physiology': asdict(brain.config),
        'execution': brain.execution_summary(),
        'mapping_profile': sim.bridge.mapping_profile,
        'body_parameters': asdict(sim.body.parameters),
        'environment': TrialEnvironment(mode=sim.environment_mode, odor=sim.odor, intensity=sim.intensity,
                                       source=tuple(float(x) for x in sim.source), senses=sim.sensors.settings, scene_id=sim.body.scene_id).to_dict(),
        'neural': {name: getattr(brain, name).detach().cpu().clone() for name in NEURAL_TENSORS},
        'rng': brain.generator.get_state(), 'tick_index': brain.tick_index,
        'neural_time_origin_ms': brain.time_origin_ms,
        'clock': {'command_hz': sim.command_hz, 'origin_seconds': sim.clock_origin, 'origin_step': sim.clock_step_origin},
        'muscle_parameters': torch.from_numpy(sim.bridge.parameters.copy()),
        'integration_state': torch.from_numpy(state),
        'body': {name: torch.from_numpy(getattr(sim.body, name).copy()) for name in BODY_TENSORS},
        'time': sim.time, 'steps': sim.steps, 'wall_seconds': sim.wall_seconds,
        'gesture_model': getattr(sim, 'gesture_model', None),
        'gesture_adapter': getattr(sim, 'gesture_adapter', None),
        'gesture_stimulus': sim.sensors.visual_object.summary() if sim.sensors.visual_object else None,
        'selected': sim.selected, 'pulses': dict(sim.pulses), 'silenced': sorted(sim.silenced),
        'history': list(sim.history), 'display_state': sim.state.clone(), 'speed': sim.speed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.live-', delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'saved': True, 'time': sim.time, 'neurons': len(brain.ids), 'bytes': path.stat().st_size}


def load_live(path: Path, directory: Path):
    """Build/validate a replacement; a failed load cannot alter the live animal."""
    from .simulation import Simulation
    from .neuromuscular import REGION_NAMES
    try:
        p = torch.load(path, map_location='cpu', weights_only=True)
        if (p['format'] not in {'flylab-live-v1', 'flylab-live-v2', 'flylab-live-v3', 'flylab-live-v4'} or p['dynamics_version'] != PAPER_DYNAMICS_VERSION
                or p['sensory_version'] != SENSORY_VERSION or p['mujoco_version'] != mujoco.__version__):
            raise ValueError('Checkpoint engine versions differ; refusing an inexact continuation')
        config = Physiology(**p['physiology'])
        execution = p.get('execution', {'device': 'cpu', 'precision': 'torch.float64', 'kernel': 'cpu-event-f64-v1'})
        device = execution['device']
        expected = {'cpu': {('torch.float64', 'cpu-event-f64-v1')},
                    'mps': {('torch.float32', METAL_VERSION), ('torch.float16', ACTIVE_HALF_METAL_VERSION + '-mark4')}}
        if device not in expected or (execution['precision'], execution['kernel']) not in expected[device]:
            raise ValueError('Checkpoint execution precision or kernel differs')
        if config.profile != 'shiu-2024':
            raise ValueError('Checkpoint requires paper physiology')
        environment = TrialEnvironment(**p['environment'])
        sim = Simulation()
        clock = p.get('clock', {'command_hz': 50, 'origin_seconds': 0., 'origin_step': 0})
        hz, origin, origin_step = clock['command_hz'], clock['origin_seconds'], clock['origin_step']
        neural_origin = p.get('neural_time_origin_ms', 0.)
        if hz not in {50, 60} or not math.isfinite(origin) or origin < 0 or type(origin_step) is not int or origin_step < 0 or not math.isfinite(neural_origin) or neural_origin < 0:
            raise ValueError('Invalid simulation clock')
        sim.command_hz = hz
        sim.configure('connectome', directory, config, BodyParameters(**p['body_parameters']), environment.scene_id,
                      device=device, precision=execution['precision'].removeprefix('torch.'), mapping_profile=p.get('mapping_profile', LEGACY_PROFILE))
        brain = sim.full_brain
        if p['format'] == 'flylab-live-v4' and 'locomotion' not in p:
            raise ValueError('Missing walking command state')
        if 'locomotion' in p:
            sim.bridge.locomotion.load_state_dict(p['locomotion'])
        if p['graph_sha256'] != brain.manifest['graph_sha256'] or p['body_sha256'] != model_fingerprint(sim.body):
            raise ValueError('Checkpoint anatomy differs from the installed model')
        for name in NEURAL_TENSORS:
            value, expected = p['neural'][name], getattr(brain, name)
            shape_ok = isinstance(value, torch.Tensor) and value.shape == expected.shape
            if name == 'delay_queue' and isinstance(value, torch.Tensor):
                # A refined clock may still hold later signals scheduled by the
                # previous coarse clock. Preserve that complete circular queue.
                shape_ok = value.ndim == 2 and value.shape[1] == len(brain.ids) and value.shape[0] >= expected.shape[0]
            if not isinstance(value, torch.Tensor) or not shape_ok or value.dtype != expected.dtype or not torch.isfinite(value).all():
                raise ValueError(f'Invalid complete neural state: {name}')
        ticks, steps = p['tick_index'], p['steps']
        if type(ticks) is not int or type(steps) is not int or min(ticks, steps) < 0 or origin_step > steps or not math.isclose(neural_origin + ticks * config.dt_ms, p['time'] * 1000, abs_tol=1e-8) or not math.isclose(p['time'], origin + (steps - origin_step) / hz, abs_tol=1e-9):
            raise ValueError('Neural and body clocks must agree')
        if not math.isfinite(p['wall_seconds']) or p['wall_seconds'] < 0 or p['speed'] not in {1, 2, 4}:
            raise ValueError('Invalid timing state')
        if p['selected'] not in REGION_NAMES or not set(p['silenced']).issubset(REGION_NAMES):
            raise ValueError('Invalid region selection')
        for region, (amplitude, expires) in p['pulses'].items():
            if region not in REGION_NAMES or not 0 <= amplitude <= 3 or not math.isfinite(expires):
                raise ValueError('Invalid intervention')
        sim.bridge.set_parameters(p['muscle_parameters'].numpy())
        gain_dtype = torch.float16 if execution['precision'] == 'torch.float16' or any(
            h.get('to_precision') == 'torch.float16' for h in execution.get('history', [])) else brain.dtype
        if not torch.equal(brain.output_gain.cpu().to(gain_dtype), p['neural']['output_gain'].to(gain_dtype)):
            raise ValueError('Neural gains disagree with checkpoint parameters')
        for name in NEURAL_TENSORS:
            setattr(brain, name, p['neural'][name].to(device=device).clone())
        brain.execution_history = execution.get('history', [])
        brain.generator.set_state(p['rng'])
        brain.time_origin_ms = neural_origin
        brain.tick_index, brain.time_ms = ticks, neural_origin + ticks * config.dt_ms
        brain.total_spikes = int(brain.spike_counts.sum())
        state = p['integration_state']
        if state.shape != (mujoco.mj_stateSize(sim.body.model, STATE_SPEC),) or state.dtype != torch.float64 or not torch.isfinite(state).all():
            raise ValueError('Invalid MuJoCo integration state')
        mujoco.mj_setState(sim.body.model, sim.body.data, state.numpy(), STATE_SPEC)
        # Rebuild derived contacts/geometry, then restore integration inputs and
        # warm-start values which forward dynamics may otherwise overwrite.
        mujoco.mj_forward(sim.body.model, sim.body.data)
        mujoco.mj_setState(sim.body.model, sim.body.data, state.numpy(), STATE_SPEC)
        if not math.isclose(sim.body.data.time, p['time'], abs_tol=1e-9):
            raise ValueError('Physical integration clock differs from checkpoint time')
        for name in BODY_TENSORS:
            value = p['body'][name]
            if value.shape != getattr(sim.body, name).shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid body state: {name}')
            setattr(sim.body, name, value.numpy().copy())
        display = p['display_state']
        if display.shape != sim.state.shape or not torch.isfinite(display).all():
            raise ValueError('Invalid display state')
        sim.state = display
        sim.time, sim.steps, sim.wall_seconds = p['time'], steps, p['wall_seconds']
        sim.clock_origin, sim.clock_step_origin = origin, origin_step
        sim.environment_mode, sim.odor, sim.intensity = environment.mode, environment.odor, environment.intensity
        sim.source, sim.sensors = np.array(environment.source), SensorSuite(environment.senses)
        sim.selected, sim.pulses, sim.silenced = p['selected'], p['pulses'], set(p['silenced'])
        sim.history, sim.speed = p['history'], p['speed']
        coupling = p.get('coupling', {'version': COUPLING_VERSION, 'mode': 'serial'})
        mode = coupling['mode']
        if coupling['version'] != COUPLING_VERSION or mode not in {'serial', 'pipelined'} or (p['format'] not in {'flylab-live-v3', 'flylab-live-v4'} and (p['format'] == 'flylab-live-v2') != (mode == 'pipelined')):
            raise ValueError('Invalid coupling mode or checkpoint version')
        sim.set_coupling(mode)
        if mode == 'pipelined':
            command = coupling['command']
            if not isinstance(command, torch.Tensor) or command.shape != (sim.body.model.nu,) or command.dtype != torch.float32 or not torch.isfinite(command).all() or torch.any((command < 0) | (command > 1)):
                raise ValueError('Invalid buffered muscle command')
            generated, source, applied, sensed = (coupling[k] for k in ['generated_at', 'source_time', 'applied_at', 'sensed_at'])
            if not isinstance(generated, (int, float)) or not math.isfinite(generated) or not math.isclose(generated, sim.time, abs_tol=1e-9):
                raise ValueError('Buffered command clock differs from simulation')
            if any(x is not None and (not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= generated) for x in [source, applied, sensed]):
                raise ValueError('Invalid buffered command timestamps')
            sim.pending_command = command.numpy().copy()
            sim.command_generated_at, sim.command_source_time = generated, source
            sim.applied_command_generated_at, sim.last_sensory_time = applied, sensed
        if p['format'] == 'flylab-live-v3' or (p['format'] == 'flylab-live-v4' and p.get('dopamine') is not None):
            brain.reward_circuit.load_state(p['dopamine'], sim.time)
        elif p.get('dopamine') is not None:
            raise ValueError('Dopamine state requires a v3 checkpoint')
        sim.running = False
        if p.get('gesture_adapter'):
            from .connectome_adapter import ConnectomeAdapter
            adapter = ConnectomeAdapter(sim.full_brain, p['gesture_adapter']['rank'],
                                        grouping=p['gesture_adapter'].get('grouping', 'superclass-side-v1'))
            adapter.apply(p['gesture_adapter']['parameters'].numpy())
            sim._gesture_adapter = adapter
            sim.gesture_adapter = p['gesture_adapter']
            sim.gesture_model = p.get('gesture_model')
        if p.get('gesture_stimulus'):
            from .gesture_scene import HandStimulus
            stimulus = p['gesture_stimulus']
            sim.sensors.visual_object = HandStimulus(stimulus['gesture'], placement=stimulus)
        return sim
    except (KeyError, TypeError, AttributeError, RuntimeError, OverflowError, pickle.UnpicklingError) as exc:
        raise ValueError('Invalid or incompatible live checkpoint') from exc
