"""Isolated aerodynamic experiments; never replace the user's loaded policy."""
import asyncio
from dataclasses import asdict
import threading
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
import mujoco
import numpy as np

from .flight import FlightRig, FlightSettings, TRIM_FREQUENCY, TRIM_AMPLITUDE, average_wingbeats
from .flight_control import free_rollout, SPINUP_SECONDS, CONTROLLER_VERSION

router = APIRouter(prefix='/api/flight')
experiment_lock = threading.Lock()
pending_experiments = set()


class FlightRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    tethered: bool = True
    air: bool = True
    drive: bool = True
    frequency: float = Field(default=TRIM_FREQUENCY, ge=150., le=300.)
    amplitude: float = Field(default=TRIM_AMPLITUDE, ge=.2, le=1.35)
    duration: Literal[.04, .2, 1.] = .04
    controller: Literal['fixed', 'hover'] = 'fixed'

    @model_validator(mode='after')
    def validate_controller(self):
        if self.controller == 'hover' and (self.tethered or not self.drive
                or self.frequency != TRIM_FREQUENCY or self.amplitude != TRIM_AMPLITUDE):
            raise ValueError('Hover assist requires free release and the calibrated wingbeat')
        return self


@router.get('/configuration')
def configuration():
    return dict(frequency=TRIM_FREQUENCY, amplitude=TRIM_AMPLITUDE,
                controller=CONTROLLER_VERSION, spinup_seconds=SPINUP_SECONDS)


def run_experiment(request):
    from .simulation import Simulation
    started = time.perf_counter()
    settings = FlightSettings(tethered=True,
        air_density=1.225e-6 if request.air else 0., air_viscosity=1.81e-5 if request.air else 0.)
    rig = FlightRig(settings)
    scaffolding = Simulation()
    scaffolding.controller = 'posture'
    scaffolding.body = rig.body
    base = scaffolding.snapshot()
    base['model'].update(name='Wing aerodynamic test rig', controller='posture',
        precision='float64', engine='MuJoCo', assumptions=rig.description()['controller'])
    base['body']['anatomy'].update(name='Experimental wing rig', muscles=196, mass_mg=rig.mass_g*1000,
                                  provenance=rig.description()['fluid'])
    base['gesture_stimulus'] = None
    frames = []
    steps = round(request.duration/settings.timestep)
    stride = max(5, int(np.ceil(steps/400)))

    def capture(sample, index):
        if (index+1) % stride == 0 or index == steps-1:
            # Recompute transforms for the new integrated pose, only at rendered
            # samples. Forces used in physics remain the native per-step forces.
            mujoco.mj_kinematics(rig.model, rig.data)
            body = rig.body.snapshot()
            dynamic = {k: body[k] for k in ('bodies', 'pretarsi', 'additional_geometry', 'feet',
                'foot_contacts', 'position', 'heading', 'upright', 'contacts')}
            frames.append(dict(time=sample['time'], body=dynamic, metrics=sample))

    if request.tethered:
        samples = rig.flap(request.duration, frequency=request.frequency if request.drive else 0.,
            amplitude=request.amplitude if request.drive else 0., feather=.8 if request.drive else 0.,
            publish=capture)
    else:
        samples = free_rollout(rig, request.duration, assist=request.controller == 'hover',
            frequency=request.frequency, amplitude=request.amplitude, drive=request.drive, publish=capture)
    means = average_wingbeats(samples, request.frequency if request.drive else 0.)
    max_tilt = float(np.degrees(np.arccos(np.clip(min(s['upright'] for s in samples), -1, 1))))
    height_error = max(abs(s['position_mm'][2]-settings.initial_height) for s in samples)
    drift = max(float(np.linalg.norm(s['position_mm'][:2])) for s in samples)
    floor_contacts = max(s['environment_contacts'] for s in samples)
    passed = max_tilt <= 15 and height_error <= 1 and drift <= 5 and floor_contacts == 0
    return dict(base=base, frames=frames,
        settings=dict(asdict(settings), tethered=request.tethered), description=rig.description(),
        controller=CONTROLLER_VERSION if request.controller == 'hover' else 'fixed-wingbeat',
        summary=dict(**means,
            weight_uN=samples[-1]['weight_uN'],
            lift_weight_ratio=float(means['mean_force_uN'][2]/samples[-1]['weight_uN']),
            simulated_seconds=request.duration, warmup_seconds=0. if request.tethered else SPINUP_SECONDS,
            wall_seconds=time.perf_counter()-started, physics_wall_seconds=rig.elapsed_wall,
            max_tilt_degrees=max_tilt, max_height_error_mm=height_error, max_horizontal_drift_mm=drift,
            environment_contacts=floor_contacts, max_wing_contacts=max(s['wing_contacts'] for s in samples),
            flight_check=None if request.tethered else ('passed' if passed else 'failed'),
            final=samples[-1]))


def _run_locked(request):
    try:
        return run_experiment(request)
    finally:
        experiment_lock.release()


@router.post('/experiment')
async def experiment(request: FlightRequest):
    if not experiment_lock.acquire(blocking=False):
        raise HTTPException(409, 'An aerodynamic experiment is already running')
    task = asyncio.create_task(asyncio.to_thread(_run_locked, request))
    pending_experiments.add(task)
    def finished(done):
        pending_experiments.discard(done)
        if not done.cancelled():
            done.exception()  # Retrieve failures even if the client disconnected.
    task.add_done_callback(finished)
    try:
        return await asyncio.shield(task)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
