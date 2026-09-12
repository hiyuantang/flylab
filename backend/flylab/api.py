"""Local, single-workbench API. Bind to loopback; not a multi-user service."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .simulation import Simulation
from .training import Trainer, evaluate
from .connectome import ConnectomeProbe

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data"
torch.set_num_threads(1)
sim = Simulation()
trainer = Trainer(DATA/"checkpoints")
failure = None


async def tick():
    global failure
    while True:
        started = asyncio.get_running_loop().time()
        try:
            if sim.running:
                sim.advance(2*sim.speed)
        except Exception as exc:
            sim.running = False
            failure = str(exc)
        await asyncio.sleep(max(.001, .04 - (asyncio.get_running_loop().time() - started)))


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(tick())
    yield
    trainer.stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="FlyLab", lifespan=lifespan)


class Command(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    action: Literal["run", "pause", "step", "reset", "stimulate", "silence", "select", "odor", "speed", "apply", "load", "restore", "environment"]
    region: Literal["optic", "antennal", "mushroom", "descending", "vnc", "motor"] = "mushroom"
    amplitude: float = Field(default=1, ge=0, le=3)
    duration: float = Field(default=.5, ge=.02, le=5)
    enabled: bool = True
    odor: Literal["A", "B", "none"] = "A"
    intensity: float = Field(default=.7, ge=0, le=1)
    speed: Literal[1, 2, 4] = 1
    checkpoint: str = ""
    environment: Literal["uniform", "spatial"] = "uniform"


@app.get("/api/health")
def health():
    return {"status": "ok", "torch": torch.__version__, "engine": "PyTorch + MuJoCo", "device": "cpu"}


@app.get("/api/state")
async def state():
    return sim.snapshot()


@app.post("/api/control")
async def control(cmd: Command):
    global failure
    try:
        match cmd.action:
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
            case "apply": sim.running = False; trainer.apply(sim)
            case "load":
                sim.running = False
                trainer.load(cmd.checkpoint, sim)
                with trainer.lock:
                    trainer.status["applied"] = cmd.checkpoint == trainer.status["checkpoint"]
            case "restore":
                from .neural import ReferenceBrain
                sim.running = False
                sim.brain = ReferenceBrain()
                sim.reset()
                with trainer.lock:
                    trainer.status["applied"] = False
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return sim.snapshot()


class TrainingRequest(BaseModel):
    episodes: int = Field(default=100, ge=10, le=1000)
    reward_odor: Literal["A", "B"] = "A"
    seed: int = Field(default=42, ge=0, le=2**31-1)


@app.post("/api/training/start")
async def train(request: TrainingRequest):
    try:
        trainer.start(sim.brain, request.episodes, request.reward_odor, request.seed)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return trainer.snapshot()


@app.post("/api/training/stop")
async def stop_training():
    trainer.stop_event.set()
    return {"stopping": True}


@app.get("/api/training")
def training():
    return {**trainer.snapshot(), "checkpoints": trainer.checkpoints()}


@app.get("/api/evaluate")
async def evaluation(reward_odor: Literal["A", "B"] = "A"):
    return evaluate(sim.brain, reward_odor)


@app.get("/api/data")
def dataset():
    path = DATA/"manifest.json"
    return json.loads(path.read_text()) if path.exists() else {"dataset": "male-cns:v1.0", "available": False, "message": "Run the connectome importer to prepare a measured subgraph."}


class ProbeRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    body_id: int = 10001
    amplitude: float = Field(default=2, ge=0, le=5)
    duration_ms: int = Field(default=100, ge=1, le=500)


@app.post("/api/probe")
def probe(request: ProbeRequest):
    try:
        return ConnectomeProbe(DATA).pulse(request.body_id, request.amplitude, request.duration_ms)
    except (ValueError, FileNotFoundError) as exc:
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
            await socket.send_json({"simulation": sim.snapshot(), "training": trainer.snapshot(), "error": failure})
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
