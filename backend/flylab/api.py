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
from dataclasses import replace
from .body import BodyParameters, FlyBody
from .pretarsus import migrate_body
from .physical_training import PhysicalTrainer
from .senses import SensorySettings, SENSORY_VERSION
from .scenes import SCENE_IDS, get_scene, scene_summary
from .physical_policy import TrialEnvironment
from .live_state import save_live, load_live
from .paper_dynamics import PAPER_DYNAMICS_VERSION
from .paper_experiment import PaperExperimentRunner
from .motor_mapping import MAPPING_PROFILE, LEGACY_PROFILE, PRETARSAL_PROFILE, PERIPHERAL_PROFILE
from .neuromuscular import NeuromuscularBridge

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data"
torch.set_num_threads(1)
sim = Simulation()
trainer = Trainer(DATA/"checkpoints")
failure = None
ready = False
workbench_lock = threading.RLock()
physical = PhysicalTrainer(DATA / "physical-checkpoints", DATA / "full")
paper = PaperExperimentRunner(DATA / 'paper-reference')


def locked(fn, *args, **kwargs):
    with workbench_lock:
        return fn(*args, **kwargs)


def advance_if_running():
    if ready and sim.running:
        sim.advance(1)


async def tick():
    global failure
    while True:
        started = asyncio.get_running_loop().time()
        try:
            if sim.running:
                # One fixed neural/body interval, irrespective of wall-clock lag.
                # Never catch up by skipping steps or increasing neural dt.
                await asyncio.to_thread(locked, advance_if_running)
        except Exception as exc:
            sim.running = False
            failure = str(exc)
        await asyncio.sleep(max(.001, .02 / sim.speed - (asyncio.get_running_loop().time() - started)))


def initialize_workbench():
    global sim, ready, failure
    try:
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
    snapshot = sim.snapshot()
    snapshot['model']['ready'] = ready
    if not ready:
        snapshot['model'].update(name='Measured brain unavailable', controller='connectome', neurons=0,
                                 edges=0, measured_connectome=False)
        snapshot['neurons'] = []
    return snapshot


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(locked, initialize_workbench)
    task = asyncio.create_task(tick())
    yield
    sim.running = False
    trainer.stop_event.set()
    physical.stop_event.set()
    paper.stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Acquiring the workbench lock also waits for an in-flight worker to finish.
    if ready:
        await asyncio.to_thread(locked, save_live, sim, DATA / 'live-state.pt')


app = FastAPI(title="FlyLab", lifespan=lifespan)


class SensingRequest(BaseModel):
    vision_model: Literal["legacy-grid-v2", "compound-retina-v1"] = "legacy-grid-v2"
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
    action: Literal["run", "pause", "step", "reset", "stimulate", "silence", "select", "odor", "speed", "environment", "controller", "physical_load", "sensing", "scene", "live_save", "live_restore", "vision_upgrade"]
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
    return {'current': sim.full_brain.execution_summary() if ready else None,
            'available': {'cpu': True, 'mps': torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader')},
            'scope': 'Live full-connectome brain on selected device; MuJoCo, sensors and training workers run on CPU.'}


@app.get('/api/execution')
def get_execution():
    return locked(execution_status)


class ExecutionRequest(BaseModel):
    device: Literal['cpu', 'mps']


@app.post('/api/execution')
def set_execution(request: ExecutionRequest):
    try:
        with workbench_lock:
            if not ready:
                raise ValueError('Load the full measured brain first')
            sim.running = False
            if sim.full_brain.voltage.device.type != request.device:
                # Preserve the original precision and pending events before any
                # conversion; never silently overwrite the sole reference state.
                backup = DATA / 'execution-backups' / f'live-{time.time_ns()}.pt'
                save_live(sim, backup)
                sim.full_brain.set_device(request.device)
                save_live(sim, DATA / 'live-state.pt')
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
    return await asyncio.to_thread(locked, current_snapshot)


@app.post("/api/control")
async def control(cmd: Command):
    return await asyncio.to_thread(locked, execute_control, cmd)


def execute_control(cmd: Command):
    global failure, sim, ready
    try:
        if not ready and cmd.action not in {'controller', 'live_restore'}:
            raise ValueError('Load the full MaleCNS graph before starting; no reduced-circuit fallback is used')
        match cmd.action:
            case 'live_save': save_live(sim, DATA / 'live-state.pt')
            case 'live_restore':
                replacement = load_live(DATA / 'live-state.pt', DATA / 'full')
                replacement.episode = sim.episode + 1
                sim, ready, failure = replacement, True, None
            case 'scene': sim.configure_scene(cmd.scene_id)
            case 'vision_upgrade':
                if not ready:
                    raise ValueError('Load the measured brain first')
                sim.running = False
                if sim.sensors.settings.vision_model != 'compound-retina-v1':
                    from .senses import SensorSuite
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
                sim.configure_senses(SensorySettings(**cmd.senses.model_dump()))
            case "controller":
                sim.configure('connectome', DATA / 'full', Physiology.paper())
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
    return current_snapshot()


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
    profile: Literal['leg-routing-v1', 'muscle-routing-v2', 'muscle-routing-v3', 'muscle-routing-v4'] = MAPPING_PROFILE


@app.post('/api/connectome/mapping')
def set_mapping(request: MappingRequest):
    try:
        with workbench_lock:
            if not ready:
                raise ValueError('Load the full measured brain first')
            sim.running = False
            if sim.bridge.mapping_profile != request.profile:
                save_live(sim, DATA / 'mapping-backups' / f'live-{time.time_ns()}.pt')
                previous, gains, previous_body = sim.bridge, sim.full_brain.output_gain, sim.body
                try:
                    appendage_model = 'peripheral-v1' if request.profile == PERIPHERAL_PROFILE else 'pretarsal-v1' if request.profile == PRETARSAL_PROFILE else 'baseline'
                    body = sim.body
                    if body.parameters.appendage_model != appendage_model:
                        body = migrate_body(body, FlyBody(replace(body.parameters, appendage_model=appendage_model), body.scene_id))
                    bridge = NeuromuscularBridge(sim.full_brain, request.profile)
                    bridge.parameters = previous.parameters.copy()
                    sim.full_brain.output_gain = gains
                    sim.bridge, sim.body = bridge, body
                    save_live(sim, DATA / 'live-state.pt')
                except Exception:
                    sim.bridge, sim.full_brain.output_gain, sim.body = previous, gains, previous_body
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
    try:
        while True:
            snapshot = await asyncio.to_thread(locked, current_snapshot)
            await socket.send_json({"simulation": snapshot, "training": trainer.snapshot(), "error": failure})
            await asyncio.sleep(.05)
    except (WebSocketDisconnect, RuntimeError):
        pass


MODELS = ROOT/"frontend"/"public"/"models"
if MODELS.exists():
    app.mount("/models", StaticFiles(directory=MODELS), name="models")

DIST = ROOT/"frontend"/"dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST/"assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST/"index.html")
