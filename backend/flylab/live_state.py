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
from .full_connectome import Physiology
from .paper_dynamics import PAPER_DYNAMICS_VERSION
from .metal_dynamics import METAL_VERSION, NEURAL_TENSORS
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
    if (not math.isclose(brain.time_ms, sim.time * 1000., abs_tol=1e-8)
            or not math.isclose(sim.body.data.time, sim.time, abs_tol=1e-9)
            or any(not torch.isfinite(getattr(brain, name)).all() for name in NEURAL_TENSORS)):
        raise ValueError('Refusing to overwrite a saved life state with an incomplete or invalid step')
    state = np.empty(mujoco.mj_stateSize(sim.body.model, STATE_SPEC))
    mujoco.mj_getState(sim.body.model, sim.body.data, state, STATE_SPEC)
    if not np.isfinite(state).all():
        raise ValueError('Cannot save non-finite physical state')
    payload = {
        'format': 'flylab-live-v1', 'dynamics_version': brain.dynamics_version,
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
        'muscle_parameters': torch.from_numpy(sim.bridge.parameters.copy()),
        'integration_state': torch.from_numpy(state),
        'body': {name: torch.from_numpy(getattr(sim.body, name).copy()) for name in BODY_TENSORS},
        'time': sim.time, 'steps': sim.steps, 'wall_seconds': sim.wall_seconds,
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
        if (p['format'] != 'flylab-live-v1' or p['dynamics_version'] != PAPER_DYNAMICS_VERSION
                or p['sensory_version'] != SENSORY_VERSION or p['mujoco_version'] != mujoco.__version__):
            raise ValueError('Checkpoint engine versions differ; refusing an inexact continuation')
        config = Physiology(**p['physiology'])
        execution = p.get('execution', {'device': 'cpu', 'precision': 'torch.float64', 'kernel': 'cpu-event-f64-v1'})
        device = execution['device']
        expected = {'cpu': ('torch.float64', 'cpu-event-f64-v1'), 'mps': ('torch.float32', METAL_VERSION)}
        if device not in expected or (execution['precision'], execution['kernel']) != expected[device]:
            raise ValueError('Checkpoint execution precision or kernel differs')
        if config.profile != 'shiu-2024':
            raise ValueError('Checkpoint requires paper physiology')
        environment = TrialEnvironment(**p['environment'])
        sim = Simulation()
        sim.configure('connectome', directory, config, BodyParameters(**p['body_parameters']), environment.scene_id,
                      device=device, mapping_profile=p.get('mapping_profile', LEGACY_PROFILE))
        brain = sim.full_brain
        if p['graph_sha256'] != brain.manifest['graph_sha256'] or p['body_sha256'] != model_fingerprint(sim.body):
            raise ValueError('Checkpoint anatomy differs from the installed model')
        for name in NEURAL_TENSORS:
            value, expected = p['neural'][name], getattr(brain, name)
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
                raise ValueError(f'Invalid complete neural state: {name}')
        ticks, steps = p['tick_index'], p['steps']
        if type(ticks) is not int or type(steps) is not int or min(ticks, steps) < 0 or not math.isclose(ticks * config.dt_ms, steps * 20.) or not math.isclose(p['time'], steps * .02):
            raise ValueError('Neural and body clocks must agree')
        if not math.isfinite(p['wall_seconds']) or p['wall_seconds'] < 0 or p['speed'] not in {1, 2, 4}:
            raise ValueError('Invalid timing state')
        if p['selected'] not in REGION_NAMES or not set(p['silenced']).issubset(REGION_NAMES):
            raise ValueError('Invalid region selection')
        for region, (amplitude, expires) in p['pulses'].items():
            if region not in REGION_NAMES or not 0 <= amplitude <= 3 or not math.isfinite(expires):
                raise ValueError('Invalid intervention')
        sim.bridge.set_parameters(p['muscle_parameters'].numpy())
        if not torch.equal(brain.output_gain.cpu(), p['neural']['output_gain']):
            raise ValueError('Neural gains disagree with checkpoint parameters')
        for name in NEURAL_TENSORS:
            setattr(brain, name, p['neural'][name].to(device=device).clone())
        brain.execution_history = execution.get('history', [])
        brain.generator.set_state(p['rng'])
        brain.tick_index, brain.time_ms = ticks, ticks * config.dt_ms
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
        sim.environment_mode, sim.odor, sim.intensity = environment.mode, environment.odor, environment.intensity
        sim.source, sim.sensors = np.array(environment.source), SensorSuite(environment.senses)
        sim.selected, sim.pulses, sim.silenced = p['selected'], p['pulses'], set(p['silenced'])
        sim.history, sim.speed = p['history'], p['speed']
        sim.running = False
        return sim
    except (KeyError, TypeError, AttributeError, RuntimeError, OverflowError, pickle.UnpicklingError) as exc:
        raise ValueError('Invalid or incompatible live checkpoint') from exc
