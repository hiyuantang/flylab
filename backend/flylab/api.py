"""Local, single-workbench API. Bind to loopback; not a multi-user service."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import torch
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .simulation import Simulation
from .training import Trainer, evaluate
from .connectome import ConnectomeProbe
from .full_connectome import Physiology, DYNAMICS_VERSION
from dataclasses import asdict, replace
from functools import lru_cache
from .body import BodyParameters, FlyBody
from .pretarsus import migrate_body, migrate_motor_command
from .physical_training import PhysicalTrainer
from .senses import SensorySettings, SensorSuite, SENSORY_VERSION
from .scenes import SCENE_IDS, get_scene, scene_summary
from .physical_policy import TrialEnvironment
from .live_state import save_live, load_live
from .paper_dynamics import PAPER_DYNAMICS_VERSION
from .paper_experiment import PaperExperimentRunner
from .motor_mapping import MAPPING_PROFILE, LEGACY_PROFILE, PRETARSAL_PROFILE, PERIPHERAL_PROFILE
from .motor_mapping import body_model_for_profile
from .neuromuscular import NeuromuscularBridge
from .scheduler import CycleScheduler
from .gesture_training import GestureTrainer, GestureSession, settings_from_simulation
from .gesture_batch import BatchGestureTrainer, default_training_execution
from .gesture_scene import HandStimulus, random_placement
from .policy_training import PolicyTrainer, policy_versions, policy_device
from .policy_model import SensorActionTransformer
from .policy_simulation import PolicySimulation
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data"
torch.set_num_threads(1)
sim = Simulation()
trainer = Trainer(DATA/"checkpoints")
failure = None
ready = False
workbench_lock = threading.RLock()
physical = PhysicalTrainer(DATA / "physical-checkpoints", DATA / "full")
gestures = BatchGestureTrainer(DATA / 'gesture-checkpoints', DATA / 'full')
gestures.execution_lock = workbench_lock
policies = PolicyTrainer(DATA / 'policy-checkpoints')
paper = PaperExperimentRunner(DATA / 'paper-reference')
scheduler = CycleScheduler()
published_frame = None
publication_revision = 0
publication_lock = threading.Lock()


def locked(fn, *args, **kwargs):
    with workbench_lock:
        return fn(*args, **kwargs)


def advance_if_running():
    if ready and sim.running:
        sim.advance(1)
        return current_snapshot()


async def tick():
    global failure
    while True:
        try:
            if ready and sim.running:
                # One fixed neural/body interval, irrespective of wall-clock lag.
                # Never catch up by skipping steps or increasing neural dt.
                revision = publication_revision
                result = await scheduler.run(lambda: asyncio.to_thread(locked, advance_if_running), 1 / sim.command_hz / sim.speed)
                if result is not None:
                    publish_frame(result, revision)
            else:
                await asyncio.sleep(.01)
        except Exception as exc:
            sim.running = False
            failure = str(exc)
        await asyncio.sleep(0)


def initialize_workbench():
    global sim, ready, failure
    try:
        active_policy = DATA / 'active-policy.json'
        if active_policy.exists():
            sim = policies.load_simulation(json.loads(active_policy.read_text())['checkpoint'])
            policies.loaded_model = sim.gesture_model
            ready, failure = True, None
            return
        path = DATA / 'live-state.pt'
        if path.exists():
            sim = load_live(path, DATA / 'full')
        else:
            sim.configure('connectome', DATA / 'full', Physiology.paper())
        ready, failure = True, None
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        ready, failure = False, str(exc)
        sim.running = False


def current_snapshot():
    from .policy_activity import policy_activity
    activity = policy_activity(sim)
    snapshot = sim.snapshot()
    if activity is not None:
        snapshot['policy_activity'] = activity
    snapshot['body_parameters'] = asdict(sim.body.parameters)
    snapshot['model']['ready'] = ready
    if not ready:
        snapshot['model'].update(name='Measured brain unavailable', controller='connectome', neurons=0,
                                 edges=0, measured_connectome=False)
        snapshot['neurons'] = []
    return snapshot


def publish_frame(snapshot, expected_revision=None):
    global published_frame, publication_revision
    with publication_lock:
        if expected_revision is not None and expected_revision != publication_revision:
            return snapshot
        published_frame = snapshot
        publication_revision += 1
    return snapshot


def with_performance(snapshot):
    return {**snapshot, 'timing': {**snapshot['timing'], 'performance': scheduler.snapshot()}}


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(locked, initialize_workbench)
    scheduler.reset()
    publish_frame(await asyncio.to_thread(locked, current_snapshot))
    task = asyncio.create_task(tick())
    yield
    sim.running = False
    trainer.stop_event.set()
    physical.stop_event.set()
    gestures.stop_event.set()
    policies.stop_event.set()
    policies.resume_event.set()
    paper.stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Acquiring the workbench lock also waits for an in-flight worker to finish.
    if ready and sim.controller == 'connectome':
        await asyncio.to_thread(locked, save_live, sim, DATA / 'live-state.pt')


app = FastAPI(title="FlyLab", lifespan=lifespan)
from .flight_api import router as flight_router
app.include_router(flight_router)


class SensingRequest(BaseModel):
    vision_model: Literal["legacy-grid-v2", "compound-retina-v1", "compound-retina-balanced-v1", "compound-retina-balanced-v2", "compound-retina-balanced-v3", "compound-retina-balanced-v4", "compound-retina-balanced-v5"] = "legacy-grid-v2"
    proprioception_model: Literal['legacy-position-v1', 'feco-opponent-v1'] = 'legacy-position-v1'
    contact_model: Literal['legacy-leg-v1', 'tarsal-contact-v1'] = 'legacy-leg-v1'
    spatial_model: Literal['legacy-v2', 'geometry-v3'] = 'legacy-v2'
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    vision_enabled: bool = True
    hearing_enabled: bool = False
    wind_enabled: bool = False
    touch_enabled: bool = True
    proprioception_enabled: bool = True
    illumination: float = Field(default=1, ge=0, le=1)
    sound_amplitude: float = Field(default=.8, ge=0, le=1)
    sound_frequency: float = Field(default=250, ge=50, le=1000)
    wind_speed: float = Field(default=20, ge=0, le=100)
    wind_direction: float = Field(default=0, ge=-180, le=180)
    stimulus_position: tuple[float, float, float] = (4., 2., 1.5)


class Command(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    action: Literal["run", "pause", "step", "reset", "stimulate", "silence", "select", "odor", "speed", "environment", "controller", "physical_load", "sensing", "scene", "live_save", "live_restore", "vision_upgrade", "reward", "reward_stop", "walk", "walk_stop", "proprioception_upgrade", "standing_trial", "spatial_upgrade"]
    target: Literal['DNg100', 'DNb08'] = 'DNg100'
    score: float = Field(default=0, ge=-1, le=1)
    episode: int | None = Field(default=None, ge=0)
    scene_id: str = "lab"
    senses: SensingRequest | None = None
    region: Literal["optic", "antennal", "mushroom", "descending", "vnc", "motor"] = "mushroom"
    amplitude: float = Field(default=1, ge=0, le=3)
    duration: float = Field(default=.5, ge=.02, le=5)
    enabled: bool = True
    odor: Literal["A", "B", "none"] = "A"
    intensity: float = Field(default=.7, ge=0, le=1)
    speed: Literal[1, 2, 4] = 1
    checkpoint: str = ""
    environment: Literal["uniform", "spatial"] = "uniform"
    controller: Literal["connectome"] = "connectome"


@app.get("/api/health")
def health():
    with workbench_lock:
        brain = sim.full_brain
        return {"status": "ok" if ready else "data-required", "torch": torch.__version__, "engine": "PyTorch + MuJoCo", "device": brain.voltage.device.type if brain else None, 'precision': str(brain.voltage.dtype) if brain else None, 'ready': ready}


def execution_status():
    return {'current': {**sim.full_brain.execution_summary(), 'muscle_command_hz': sim.command_hz, **sim.coupling_summary()} if ready and sim.full_brain is not None else None,
            'available': {'cpu': True, 'mps': torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader')},
            'scope': 'Live full-connectome brain on selected device; MuJoCo and sensors run on CPU. Each training controller reports its own device.'}


@app.get('/api/execution')
def get_execution():
    return locked(execution_status)


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    device: Literal['cpu', 'mps']
    precision: Literal['float16', 'float32', 'float64'] | None = None
    coupling_mode: Literal['serial', 'pipelined'] | None = None
    neural_dt_ms: Literal[.1, 1.] | None = None
    muscle_command_hz: Literal[50, 60] | None = None


@app.post('/api/execution')
def set_execution(request: ExecutionRequest):
    try:
        with workbench_lock:
            if not ready or sim.controller != 'connectome':
                raise ValueError('Execution controls here require the measured connectome')
            sim.running = False
            brain = sim.full_brain
            hz = request.muscle_command_hz or sim.command_hz
            if (request.coupling_mode is not None and request.coupling_mode != sim.coupling_mode) or brain.voltage.device.type != request.device or (request.precision is not None and str(brain.dtype) != f'torch.{request.precision}') or hz != sim.command_hz or (request.neural_dt_ms is not None and request.neural_dt_ms != brain.config.dt_ms):
                # Preserve the original precision and pending events before any
                # conversion; never silently overwrite the sole reference state.
                backup = DATA / 'execution-backups' / f'live-{time.time_ns()}.pt'
                save_live(sim, backup)
                from .metal_dynamics import NEURAL_TENSORS
                names = (*NEURAL_TENSORS, 'device', 'dtype', 'metal', 'config', 'delay_steps', 'tick_index', 'time_origin_ms')
                previous = {name: getattr(brain, name) for name in names}
                history = list(brain.execution_history)
                clock = (sim.command_hz, sim.clock_origin, sim.clock_step_origin)
                coupling_fields = ['coupling_mode', 'pending_command', 'command_generated_at', 'command_source_time', 'applied_command_generated_at', 'last_sensory_time', 'cycle_incomplete']
                coupling_previous = {key: getattr(sim, key) for key in coupling_fields}
                try:
                    brain.set_device(request.device, request.precision)
                    if request.neural_dt_ms is not None or hz != sim.command_hz:
                        sim.set_command_hz(hz, request.neural_dt_ms)
                    if request.coupling_mode is not None:
                        sim.set_coupling(request.coupling_mode)
                    save_live(sim, DATA / 'live-state.pt')
                    scheduler.reset()
                    publish_frame(current_snapshot())
                except Exception:
                    for name, value in previous.items():
                        setattr(brain, name, value)
                    brain.execution_history = history
                    sim.command_hz, sim.clock_origin, sim.clock_step_origin = clock
                    for key, value in coupling_previous.items():
                        setattr(sim, key, value)
                    raise
            return execution_status()
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get('/api/scenes')
def scenes():
    return [scene_summary(key) for key in SCENE_IDS]


@app.get('/api/scenes/{scene_id}')
def scene_definition(scene_id: str):
    try:
        return get_scene(scene_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/state")
async def state():
    snapshot = published_frame if sim.running and published_frame is not None else await asyncio.to_thread(locked, current_snapshot)
    return with_performance(snapshot)


@app.post("/api/control")
async def control(cmd: Command):
    return await asyncio.to_thread(locked, execute_control, cmd)


def execute_control(cmd: Command):
    global failure, sim, ready
    try:
        if not ready and cmd.action not in {'controller', 'live_restore'}:
            raise ValueError('Load the full MaleCNS graph before starting; no reduced-circuit fallback is used')
        if cmd.action in {'run', 'step'} and (gestures.snapshot(False)['running'] or policies.snapshot(False)['running']):
            raise ValueError('Pause training before running the experiment controller')
        if cmd.action in {'reward', 'reward_stop', 'walk', 'walk_stop'}:
            if sim.controller != 'connectome' or sim.full_brain.config.profile != 'shiu-2024' or sim.cycle_incomplete:
                raise ValueError('Neural intervention requires a complete measured paper-profile simulation cycle')
            if cmd.episode is not None and cmd.episode != sim.episode:
                raise ValueError('This neural intervention belongs to a previous experiment; refresh first')
        match cmd.action:
            case 'spatial_upgrade':
                sim.running = False
                save_live(sim, DATA / 'sensory-backups' / f'live-{time.time_ns()}.pt')
                previous = sim.sensors
                try:
                    sim.sensors = SensorSuite(replace(previous.settings, spatial_model='geometry-v3', contact_model='tarsal-contact-v1'))
                    save_live(sim, DATA / 'live-state.pt')
                except Exception:
                    sim.sensors = previous
                    raise
            case 'standing_trial':
                save_live(sim, DATA / 'standing-backups' / f'live-{time.time_ns()}.pt')
                sim.prepare_standing_trial()
            case 'walk': sim.bridge.locomotion.submit(cmd.target, cmd.amplitude, cmd.duration, sim.time)
            case 'walk_stop': sim.bridge.locomotion.reset()
            case 'proprioception_upgrade':
                # No reset or neural replacement; old checkpoints retain their
                # original sensory profile. Record the upgraded life atomically.
                sim.running = False
                save_live(sim, DATA / 'sensory-backups' / f'live-{time.time_ns()}.pt')
                previous = sim.sensors
                try:
                    sim.sensors = SensorSuite(replace(previous.settings, proprioception_model='feco-opponent-v1'))
                    save_live(sim, DATA / 'live-state.pt')
                except Exception:
                    sim.sensors = previous
                    raise
            case 'reward': sim.full_brain.reward_circuit.submit(cmd.score, sim.time)
            case 'reward_stop': sim.full_brain.reward_circuit.stop()
            case 'live_save': save_live(sim, DATA / 'live-state.pt')
            case 'live_restore':
                replacement = load_live(DATA / 'live-state.pt', DATA / 'full')
                replacement.episode = sim.episode + 1
                sim, ready, failure = replacement, True, None
                (DATA / 'active-policy.json').unlink(missing_ok=True)
            case 'scene': sim.configure_scene(cmd.scene_id)
            case 'vision_upgrade':
                if not ready:
                    raise ValueError('Load the measured brain first')
                sim.running = False
                if sim.sensors.settings.vision_model != 'compound-retina-v1':
                    save_live(sim, DATA / 'sensory-backups' / f'live-{time.time_ns()}.pt')
                    previous = sim.sensors
                    try:
                        sim.bridge.retina()
                        sim.sensors = SensorSuite(replace(previous.settings, vision_model='compound-retina-v1'))
                        save_live(sim, DATA / 'live-state.pt')
                    except Exception:
                        sim.sensors = previous
                        raise
            case 'sensing':
                if cmd.senses is None:
                    raise ValueError('Sensory settings are required')
                if cmd.senses.vision_model.startswith('compound-retina-balanced-') and not isinstance(sim, PolicySimulation):
                    raise ValueError('Bio-inspired optics require the transformer controller')
                sim.configure_senses(SensorySettings(**cmd.senses.model_dump()))
            case "controller":
                profile = Physiology.paper(dt_ms=1000 / 960, timing_rounding='ceil') if sim.command_hz == 60 else Physiology.paper()
                sim.configure('connectome', DATA / 'full', profile)
                ready, failure = True, None
            case "physical_load":
                checkpoint = physical.load(cmd.checkpoint)
                if checkpoint['mode'] != 'connectome' or checkpoint.get('physiology', {}).get('profile') != 'shiu-2024':
                    raise ValueError('This workbench requires a full-graph paper-profile policy')
                sim.load_physical(checkpoint, DATA / 'full')
            case "run": sim.running = True
            case "pause": sim.running = False
            case "step": sim.running = False; sim.advance()
            case "reset": sim.running = False; sim.reset(); failure = None
            case "environment": sim.running = False; sim.environment_mode = cmd.environment; sim.reset()
            case "select": sim.selected = cmd.region
            case "stimulate": sim.selected = cmd.region; sim.stimulate(cmd.region, cmd.amplitude, cmd.duration)
            case "silence":
                if cmd.enabled: sim.silenced.add(cmd.region)
                else: sim.silenced.discard(cmd.region)
            case "odor": sim.odor = cmd.odor; sim.intensity = cmd.intensity
            case "speed": sim.speed = cmd.speed
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if cmd.action in {'reset', 'live_restore', 'controller', 'scene'}:
        scheduler.reset()
    return with_performance(publish_frame(current_snapshot()))


class TrainingRequest(BaseModel):
    episodes: int = Field(default=100, ge=10, le=1000)
    reward_odor: Literal["A", "B"] = "A"
    seed: int = Field(default=42, ge=0, le=2**31-1)


@app.post("/api/training/start")
async def train(request: TrainingRequest):
    raise HTTPException(410, 'Synthetic-circuit training is retired from the workbench; use full-graph physical training')


@app.post("/api/training/stop")
async def stop_training():
    trainer.stop_event.set()
    return {"stopping": True}


@app.get("/api/training")
def training():
    return {**trainer.snapshot(), "checkpoints": trainer.checkpoints()}


@app.get("/api/evaluate")
async def evaluation(reward_odor: Literal["A", "B"] = "A"):
    raise HTTPException(410, 'Synthetic odor-classifier evaluation is retired')


@app.get("/api/data")
def dataset():
    path = DATA/"manifest.json"
    return json.loads(path.read_text()) if path.exists() else {"dataset": "male-cns:v1.0", "available": False, "message": "Run the connectome importer to prepare a measured subgraph."}


@app.get('/api/connectome/full')
def full_graph():
    path = DATA / 'full' / 'manifest.json'
    return {**json.loads(path.read_text()), 'available': True} if path.exists() else {'available': False}


def evidence_index():
    if not ready or sim.full_brain is None or sim.bridge is None:
        raise HTTPException(503, 'The measured brain is not loaded yet.')
    from .neuron_evidence import NeuronEvidence
    if not hasattr(sim.bridge, '_evidence_index'):
        sim.bridge._evidence_index = NeuronEvidence(sim.full_brain, sim.bridge)
    return sim.bridge._evidence_index


@app.get('/api/evidence')
def evidence_metadata():
    with workbench_lock:
        return evidence_index().metadata()


@app.get('/api/evidence/neurons')
def evidence_neurons(query: str = Query('', max_length=120),
                     superclass: str = Query('', max_length=120),
                     scope: Literal['all', 'sensory', 'motor'] = 'all',
                     offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100)):
    with workbench_lock:
        return evidence_index().page(query, superclass, scope, offset, limit)


@app.get('/api/evidence/neurons/{body_id}')
def evidence_neuron(body_id: int):
    with workbench_lock:
        try:
            return evidence_index().detail(body_id, sim.sensors.settings)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc


@app.get('/api/anatomy')
def anatomy(group: str = Query(default='all', max_length=2048),
            view: Literal['groups', 'neurons', 'wiring'] = 'groups',
            query: str = Query(default='', max_length=120),
            offset: int = Query(default=0, ge=0), limit: int = Query(default=30, ge=1, le=100)):
    with workbench_lock:
        if not ready or sim.full_brain is None:
            raise HTTPException(503, 'The measured brain is not loaded yet.')
        try:
            return sim.full_brain.anatomy.snapshot(group, view, query, offset, limit)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc


def measured_view():
    if not ready or sim.full_brain is None:
        raise HTTPException(503, 'The measured brain is not loaded yet.')
    return sim.full_brain.neural_view


@app.get('/api/neural-view')
def neural_view_metadata():
    with workbench_lock:
        return measured_view().metadata()


@app.get('/api/neural-view/layout')
def neural_view_layout():
    with workbench_lock:
        view = measured_view()
        return Response(view.layout, media_type='application/octet-stream', headers={
            'X-Graph-SHA256': view.brain.manifest['graph_sha256'], 'Cache-Control': 'no-cache'})


@app.get('/api/neural-view/activity')
def neural_view_activity():
    with workbench_lock:
        view = measured_view()
        data = view.brain.rates.detach().cpu().numpy().astype('<f4').tobytes()
        return Response(data, media_type='application/octet-stream', headers={
            'X-Neural-Time-Ms': str(view.brain.time_ms),
            'X-Graph-SHA256': view.brain.manifest['graph_sha256'], 'Cache-Control': 'no-store'})


@app.get('/api/neural-view/neuron/{body_id}')
def neural_view_neuron(body_id: int, direction: Literal['incoming', 'outgoing'] = 'incoming',
                       offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
    with workbench_lock:
        try:
            return measured_view().neuron(body_id, direction, offset, limit)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc


@app.get('/api/paper')
def paper_status():
    return paper.snapshot()


@app.post('/api/paper/start')
def start_paper():
    try:
        paper.start()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return paper.snapshot()


@app.post('/api/paper/stop')
def stop_paper():
    paper.stop_event.set()
    return paper.snapshot()


@app.get('/api/connectome/mapping')
def mapping():
    with workbench_lock:
        if sim.bridge is None:
            return {'available': False, 'message': 'Select the MaleCNS controller to load its peripheral mapping.'}
        return {'available': True, **sim.bridge.summary(), 'physiology': sim.full_brain.summary()}


class MappingRequest(BaseModel):
    profile: Literal['leg-routing-v1', 'muscle-routing-v2', 'muscle-routing-v3', 'muscle-routing-v4', 'muscle-routing-v5'] = MAPPING_PROFILE


@app.post('/api/connectome/mapping')
def set_mapping(request: MappingRequest):
    try:
        with workbench_lock:
            if not ready or sim.controller != 'connectome':
                raise ValueError('Execution controls here require the measured connectome')
            sim.running = False
            if sim.bridge.mapping_profile != request.profile:
                save_live(sim, DATA / 'mapping-backups' / f'live-{time.time_ns()}.pt')
                previous, gains, previous_body = sim.bridge, sim.full_brain.output_gain, sim.body
                previous_command = sim.pending_command
                try:
                    appendage_model = body_model_for_profile(request.profile)
                    body = sim.body
                    if body.parameters.appendage_model != appendage_model:
                        body = migrate_body(body, FlyBody(replace(body.parameters, appendage_model=appendage_model), body.scene_id))
                    bridge = NeuromuscularBridge(sim.full_brain, request.profile)
                    bridge.parameters = previous.parameters.copy()
                    bridge.locomotion.load_state_dict(previous.locomotion.state_dict())
                    sim.full_brain.output_gain = gains
                    sim.pending_command = migrate_motor_command(previous_body, body, previous_command)
                    sim.bridge, sim.body = bridge, body
                    save_live(sim, DATA / 'live-state.pt')
                except Exception:
                    sim.bridge, sim.full_brain.output_gain, sim.body = previous, gains, previous_body
                    sim.pending_command = previous_command
                    raise
            return mapping()
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get('/api/mechanics')
def mechanics():
    return locked(sim.body.mechanics)


class MechanicsRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    mass_scale: float = Field(default=1, ge=.1, le=3)
    strength_scale: float = Field(default=1, ge=.1, le=3)
    friction: float = Field(default=1, ge=.1, le=3)
    limit_mode: Literal['baseline', 'reference_envelope'] = 'baseline'


@app.post('/api/mechanics')
def set_mechanics(request: MechanicsRequest):
    try:
        with workbench_lock:
            sim.configure(sim.controller, DATA / 'full', body_parameters=replace(sim.body.parameters, **request.model_dump()))
            return sim.body.mechanics()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


class PhysiologyRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    gain: float = Field(default=.275, ge=0, le=100)
    membrane_ms: float = Field(default=20, ge=1, le=100)
    synapse_ms: float = Field(default=5, ge=1, le=100)
    glutamate_sign: Literal[-1, 0, 1] = -1


@app.post('/api/connectome/physiology')
def set_physiology(request: PhysiologyRequest):
    try:
        with workbench_lock:
            sim.configure('connectome', DATA / 'full', Physiology.paper(**request.model_dump()))
            return sim.full_brain.summary()
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


class PhysicalRequest(BaseModel):
    mode: Literal['connectome'] = 'connectome'
    task: Literal['stand', 'walk'] = 'stand'
    generations: int = Field(default=4, ge=1, le=100)
    population: int = Field(default=6, ge=4, le=32)
    horizon: int = Field(default=50, ge=5, le=500)
    seed: int = Field(default=42, ge=0, le=2**31-1)


@app.post('/api/physical/start')
def start_physical(request: PhysicalRequest):
    try:
        with workbench_lock:
            physical.start(**request.model_dump(), body_parameters=sim.body.parameters,
                           mapping_profile=sim.bridge.mapping_profile if sim.bridge else MAPPING_PROFILE,
                           physiology=sim.full_brain.config if sim.full_brain else Physiology.paper(),
                           environment=TrialEnvironment(mode=sim.environment_mode, odor=sim.odor, intensity=sim.intensity,
                                                        source=tuple(sim.source), senses=sim.sensors.settings, scene_id=sim.body.scene_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return physical.snapshot()


@app.get('/api/physical')
def physical_status():
    return {**physical.snapshot(), 'checkpoints': physical.checkpoints()}


@app.get('/api/physical/checkpoint/{name}')
def physical_checkpoint(name: str):
    try:
        checkpoint = physical.load(name)
        return {**{k: v for k, v in checkpoint.items() if k != 'parameters'},
                'evaluation_matches_current_dynamics': checkpoint['mode'] == 'connectome' and checkpoint.get('dynamics_version') == PAPER_DYNAMICS_VERSION and checkpoint.get('sensory_version') == SENSORY_VERSION and sim.bridge is not None and checkpoint.get('mapping_profile', LEGACY_PROFILE) == sim.bridge.mapping_profile}
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/api/physical/stop')
def stop_physical():
    physical.stop_event.set()
    return {'stopping': True}


class ProbeRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    body_id: int = 10001
    amplitude: float = Field(default=2, ge=0, le=5)
    duration_ms: int = Field(default=100, ge=1, le=500)


@app.post("/api/probe")
def probe(request: ProbeRequest):
    try:
        return ConnectomeProbe(DATA).pulse(request.body_id, request.amplitude, request.duration_ms)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.websocket("/ws")
async def stream(socket: WebSocket):
    # Reject cross-site browser access to the local controller.
    origin = socket.headers.get("origin", "")
    from urllib.parse import urlparse
    if origin and urlparse(origin).hostname not in {"localhost", "127.0.0.1"}:
        await socket.close(code=1008)
        return
    await socket.accept()
    last_token, last_sent = None, 0.
    try:
        while True:
            running = sim.running
            token = (publication_revision, scheduler.phase, failure, running)
            now = asyncio.get_running_loop().time()
            if not running or token != last_token or now - last_sent >= .25:
                snapshot = published_frame if running and published_frame is not None else await asyncio.to_thread(locked, current_snapshot)
                snapshot = with_performance(snapshot)
                await socket.send_json({"simulation": snapshot, "training": trainer.snapshot(), "error": failure})
                last_token, last_sent = token, now
            await asyncio.sleep(1 / 60 if running else .1)
    except (WebSocketDisconnect, RuntimeError):
        pass


from .gesture_gradients import DEFAULT_SURROGATE_SCALE


class GestureRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    iterations: int = Field(default=10, ge=1, le=1000)
    horizon: int = Field(default=25, ge=5, le=250)
    rank: int = Field(default=2, ge=1, le=64)
    batch_size: int = Field(default=3, ge=1, le=64)
    learning_rate: float = Field(default=.01, gt=0, le=.1)
    surrogate_scale: float = Field(default=DEFAULT_SURROGATE_SCALE, gt=0, le=1)
    early_stopping: bool = True
    early_stopping_patience: int = Field(default=10, ge=1, le=1000)
    proportions: dict[str, float] | None = None
    chart_name: str = Field(default='Hand gestures', min_length=1, max_length=80)
    version_name: str = Field(default='', max_length=120)
    seed: int = Field(default=42, ge=0, le=2147483647)
    gesture: Literal['palm', 'fist', 'point', 'point_right', 'point_both'] | None = None
    checkpoint: str | None = None
    resume_from: Literal['latest', 'best'] = 'latest'
    demonstrations_per_gesture: int = Field(default=16, ge=1, le=128)
    validation_episodes: int = Field(default=4, ge=1, le=32)
    rollout_episodes: int = Field(default=4, ge=1, le=32)
    evaluation_interval: int = Field(default=10, ge=1, le=1000)


def training_backend(controller_kind):
    return policies if controller_kind == 'transformer' else gestures


def combined_training_status(controller_kind='connectome'):
    backend = training_backend(controller_kind)
    value = backend.snapshot()
    value['controller_kind'] = controller_kind
    value['versions'] = gestures.snapshot(False)['versions'] + policy_versions(policies.directory)
    value['checkpoints'] = [v['id'] for v in value['versions']]
    value['loaded_model'] = getattr(sim, 'gesture_model', None)
    value['other_training_running'] = training_backend('connectome' if controller_kind == 'transformer' else 'transformer').snapshot(False)['running']
    if value.get('frame') and hasattr(backend, 'preview_body_parameters'):
        value['frame']['body_parameters'] = backend.preview_body_parameters
    return value


@app.get('/api/gestures')
def gesture_status(controller_kind: Literal['connectome', 'transformer'] = 'connectome'):
    return combined_training_status(controller_kind)


class GestureTargetRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    gesture: Literal['palm', 'fist', 'point', 'point_right', 'point_both']
    steps: int = Field(default=10, ge=1, le=250)
    parameters: BodyParameters


@lru_cache(maxsize=32)
def target_response(gesture: str, steps: int, parameters: BodyParameters):
    """The same physically replayed activation targets used by the batch trainer."""
    from .gesture_targets import muscle_reference
    reference = muscle_reference(gesture, steps, parameters)
    return {'body': reference.body.snapshot(), 'steps': steps, 'time': steps * .02,
            'selected_muscles': list(range(84)),
            'target_activation': reference.activations[-1, :84].tolist(),
            'validation': reference.evidence}


@app.post('/api/gestures/target')
async def gesture_target(request: GestureTargetRequest):
    try:
        return await asyncio.to_thread(target_response, request.gesture, request.steps, request.parameters)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post('/api/gestures/start')
async def gesture_start(request: GestureRequest, controller_kind: Literal['connectome', 'transformer'] = 'connectome'):
    backend = training_backend(controller_kind)
    def start():
        if not ready and controller_kind == 'connectome':
            raise ValueError('Full MaleCNS workbench is unavailable')
        other = policies if backend is gestures else gestures
        if other.snapshot(False)['running']:
            raise ValueError('Stop the other controller training run first')
        if physical.snapshot()['running'] or paper.snapshot()['running']:
            raise ValueError('Stop other training/reference experiments before starting a gesture run')
        sim.running = False
        settings = settings_from_simulation(sim)
        # Parent versions supply their own mechanics, independent of the live fly.
        parameters = (backend._load(request.checkpoint)['settings']['body']
                      if request.checkpoint else settings['body'])
        options = request.model_dump()
        if controller_kind != 'transformer':
            for key in ('resume_from', 'demonstrations_per_gesture', 'validation_episodes', 'rollout_episodes', 'evaluation_interval'):
                options.pop(key)
        backend.start(settings, **options)
        with backend.lock:
            backend.preview_body_parameters = dict(parameters)
    try:
        await asyncio.to_thread(locked, start)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))
    return combined_training_status(controller_kind)


@app.post('/api/gestures/stop')
def gesture_stop(controller_kind: Literal['connectome', 'transformer'] = 'connectome'):
    return training_backend(controller_kind).request_stop()


@app.post('/api/gestures/resume')
def gesture_resume(controller_kind: Literal['connectome', 'transformer'] = 'connectome'):
    try:
        if physical.snapshot()['running'] or paper.snapshot()['running']:
            raise ValueError('Stop other training/reference experiments before resuming')
        other = policies if controller_kind == 'connectome' else gestures
        if other.snapshot(False)['running']:
            raise ValueError('Stop the other controller training run first')
        return training_backend(controller_kind).resume()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


class DeleteGestureVersions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    version: str
    expected_ids: list[str]


@app.post('/api/gestures/versions/delete')
async def delete_gesture_versions(request: DeleteGestureVersions):
    try:
        return await asyncio.to_thread(locked, lambda: (policies if request.version.startswith('policy-') else gestures).delete_versions(
            request.version, request.expected_ids, getattr(sim, 'gesture_model', None)))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@app.get('/api/gestures/versions/{name}/previews/{epoch}/{sample}')
def gesture_preview(name: str, epoch: int, sample: int):
    try:
        return policies.load_preview(name, epoch, sample)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc))


@app.get('/api/gestures/versions/{name}')
def gesture_version(name: str):
    try:
        from .gesture_versions import version_metadata
        backend = policies if name.startswith('policy-') else gestures
        versions = policy_versions(policies.directory) if backend is policies else version_metadata(gestures.directory)
        metadata = next((x for x in versions if x['id'] == name), None)
        if metadata is None:
            raise ValueError('Weight version does not exist')
        saved = torch.load(backend.directory / name, map_location='cpu', weights_only=True)
        return {'metadata': metadata, 'results': saved['results']}
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


class HandRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    gesture: Literal['palm', 'fist', 'point', 'point_right', 'point_both'] | None = None


@app.post('/api/gestures/hand')
async def select_hand(request: HandRequest):
    def select():
        if not ready or (sim.controller != 'connectome' and not isinstance(sim, PolicySimulation)):
            raise ValueError('Load a connectome or transformer model before presenting a hand')
        body = sim.body.snapshot()
        sim.sensors.visual_object = (HandStimulus(request.gesture,
            placement=random_placement(np.random.default_rng(), body['position'], body['heading']))
            if request.gesture else None)
        if request.gesture:
            sim.sensors.settings = replace(sim.sensors.settings, vision_enabled=True, calibration_sphere=False,
                vision_model=sim.sensors.settings.vision_model if isinstance(sim, PolicySimulation) else 'compound-retina-v1')
            sim.running = True
        return publish_frame(current_snapshot())
    try:
        return await asyncio.to_thread(locked, select)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


class GestureModelRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    checkpoint: str | None = None


@app.post('/api/gestures/load')
async def load_gesture_model(request: GestureModelRequest, controller_kind: Literal['connectome', 'transformer'] = 'connectome'):
    def load():
        global sim, ready, failure
        if gestures.snapshot(False)['running'] or policies.snapshot(False)['running']:
            raise ValueError('Wait for training to finish before loading a model')
        if (request.checkpoint and request.checkpoint.startswith('policy-')) or (not request.checkpoint and controller_kind == 'transformer'):
            if request.checkpoint:
                replacement = policies.load_simulation(request.checkpoint)
            else:
                settings = settings_from_simulation(sim)
                body = FlyBody(BodyParameters(**settings['body']))
                model = SensorActionTransformer(action_names=[body.model.actuator(i).name for i in range(84)])
                replacement = PolicySimulation(model.to(policy_device()), settings)
            if sim.controller == 'connectome' and ready:
                save_live(sim, DATA / 'live-state.pt')
            sim = replacement
            ready, failure = True, None
            policies.loaded_model = request.checkpoint
            active = DATA / 'active-policy.json'
            if request.checkpoint:
                temporary = active.with_suffix('.tmp')
                temporary.write_text(json.dumps({'checkpoint': request.checkpoint}))
                temporary.replace(active)
            else:
                active.unlink(missing_ok=True)
            scheduler.reset()
            publish_frame(current_snapshot())
            return {'loaded_model': request.checkpoint, 'simulation': sim.snapshot()}
        saved = gestures._load(request.checkpoint) if request.checkpoint else None
        settings = saved['settings'] if saved else settings_from_simulation(sim)
        if not saved:
            settings['training_execution'] = default_training_execution()
        # Build and validate before replacing the active simulation. Loading a
        # model intentionally starts a paused lab trial with its training settings.
        session = GestureSession(DATA / 'full', settings, saved['rank'] if saved else 2,
                                 saved['seed'] if saved else 42)
        if saved:
            if saved['groups'] != [list(g) for g in session.adapter.groups]:
                raise ValueError('Saved adapter group registry differs')
            session.adapter.apply(saved['parameters'].numpy())
        session.sim.gesture_model = request.checkpoint
        session.sim._gesture_adapter = session.adapter
        session.sim.gesture_adapter = {'rank': session.adapter.rank, 'grouping': session.adapter.grouping, 'parameters': torch.from_numpy(session.adapter.parameters.copy())}
        session.sim.running = False
        sim = session.sim
        gestures.loaded_model = request.checkpoint
        (DATA / 'active-policy.json').unlink(missing_ok=True)
        ready, failure = True, None
        scheduler.reset()
        publish_frame(current_snapshot())
        return {'loaded_model': request.checkpoint, 'simulation': sim.snapshot()}
    try:
        return await asyncio.to_thread(locked, load)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


MODELS = ROOT/"frontend"/"public"/"models"
if MODELS.exists():
    app.mount("/models", StaticFiles(directory=MODELS), name="models")

DIST = ROOT/"frontend"/"dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST/"assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST/"index.html")
