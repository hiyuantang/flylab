# Simulation scheduling and smooth playback

The live workbench offers **60 Hz commands / 960 Hz brain** in Data & model → Brain execution. Rates refer to simulated time; real-time throughput is measured separately. All imported neurons and connections remain present. This explicit coarse timing changes dynamics and is not a reference-equivalent optimization.

## One command cycle

In **Serial** mode, every 1/60 simulated second:

1. Sample sensory input and hold the resulting neural drive.
2. Advance the entire brain through 16 steps at 960 Hz.
3. Read motor activity and update muscle excitation.
4. Integrate muscle/contact physics for 1/60 second.

MuJoCo retains muscle activation and velocity across commands. Excitation stays constant during an interval, while force, acceleration, velocity and contacts continue evolving. It does not enforce constant movement speed. Physics uses 0.1 ms substeps plus a smaller final substep so the interval ends exactly; it never rounds a 60 Hz cycle to 16 or 17 ms. Current inputs and time constants already account for elapsed time, with no blanket signal-strength multiplier.

The nominal 1.8 ms neural delay rounds up to 2.083 ms; the nominal 2.2 ms refractory duration rounds up to 3.125 ms. The older 50 Hz exchange with 0.1 or 1 ms neural timing remains selectable. Switching backs up and preserves the animal's time, continuous state, queued signals, counts and RNG. Clock origins allow switching away from zero without resetting the animal. Checkpoints store both clocks and retain compatibility with earlier 50 Hz checkpoints.

## Concurrent brain and body

**Data & model → Brain execution → Overlap brain & body** enables an explicit one-cycle command buffer. At each boundary `t`, senses capture the body before either branch starts:

```text
                         same simulated interval [t, t + dt]
senses at t ── GPU brain ───────────────────────────► next muscle command
held command ─ CPU MuJoCo physics ──────────────────► next body state
                                                      │
                                  join both, publish, advance shared clock
```

The CPU integrates the previously buffered excitation while the GPU computes its successor. Both retain every integration step. This adds one command interval of delay relative to serial coupling: 16.667 simulated ms at 60 Hz, or 20 ms at 50 Hz. This is an explicit modeling approximation, not an empirically calibrated nerve or muscle delay. On activation, the first interval holds current body excitation; reset starts from reset controls. Forces and velocities still evolve under physics.

Only physics runs in a bounded worker pool. The caller owns neural state, captures sensory drive before body mutation, and joins both branches before publishing or releasing the workbench lock. Neither clock can run arbitrarily ahead. Exceptions still join the physical worker; an incomplete cycle cannot advance again or overwrite a valid checkpoint. Reset or restoration is required after an incomplete cycle.

Pipeline checkpoints use `flylab-live-v2` and preserve the pending command, generation/source timestamps, physical state, neural state and clocks. Older v1 checkpoints load in serial mode. Execution changes pause and create a backup before saving the new mode. The API exposes coupling mode and added delay in `/api/execution`, plus buffer timestamps in each completed snapshot.

`tests/test_coupling.py` compares concurrent execution with a sequential reference using the exact same delayed inputs on CPU and MPS. The reference changes only wall-clock scheduling. Tests also cover command ordering, failure joins, checkpoint continuation, invalid buffers and clock changes. `scripts/benchmark_coupling.py` performs the full-graph comparison including snapshot preparation and verifies identical final neural tensors, RNG, complete MuJoCo integration state and buffered commands.

## Wall-clock deadlines

At 1× pacing, one cycle has a 16.667 ms wall-clock budget. The scheduler permits one in-flight cycle. A completed early result waits for its deadline before publication; an overdue worker finishes its complete brain/body step. It is never cancelled halfway through a step, and the scheduler never drops cycles or accumulates a catch-up queue. Slower execution therefore produces slow motion. A manual Step is an explicit unpaced inspection action.

The performance indicator shows target and achieved cycles/s, last complete cycle compute time versus its budget, and missed deadlines. Achieved rate uses up to 60 recent completed cycles, including pacing waits. Timing includes brain/body/sensory snapshot preparation, so it is not a neural-kernel benchmark or browser FPS. The indicator updates while a worker is overdue without blocking on the neural-state lock. There are no performance notifications or pop-ups.

## Display playback

The browser buffers computed poses and interpolates at its render refresh rate. It adapts playback speed to arrival cadence and retains roughly a snapshot of slack. Each limb interpolates in its parent's coordinates; composing those transforms keeps joint anchors attached. Rotations use quaternion interpolation. Meshes, extra appendages, feet, skeleton, follow camera and nearby lighting use the same displayed transforms.

Interpolation is display-only and never feeds back into physics or sensory input. It does not invent new neural activity or extrapolate motion through unknown contacts. A long stall eventually exhausts the available poses and holds the last one. Pause drains to the latest computed pose; resets/restores clear old playback. The buffer is bounded at 120 poses, and per-frame transforms use Three.js references rather than React state updates.

## Validation

Run backend checks with `PYTHONPATH=backend .venv/bin/pytest -q`, build with `(cd frontend && npm run build)`, and test display math with `node --experimental-strip-types --test scripts/test_motion.mjs` using the installed Node runtime that supports TypeScript stripping.

146 backend tests and four display tests passed. Coverage includes early deadlines, overdue workers, shutdown waiting, nonzero-time clock migration, exact saved-state continuation, fractional physics intervals, joint attachment during rotation, intermediate display poses, no extrapolation, resets and bounded buffering.

A desktop Chrome check at 1600×1100 recorded 261 distinct displayed poses from 20 computed snapshots over approximately 4.5 wall seconds. Display time never exceeded computed time. The live FP32 performance indicator reported approximately 4.2 completed cycles/s against a 60 cycles/s target on this loaded M1 Pro. This is a short interactive check with rendering active, not sustained performance evidence. Mobile Chrome at 390×844 had no horizontal overflow; neither viewport produced console errors. Long-run dynamics and biological fidelity at 960 Hz remain unvalidated.
