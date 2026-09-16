# GPU execution

The live MaleCNS brain supports three explicit execution modes:

| Mode | Neural arithmetic | Status |
|---|---|---|
| CPU | float64, sparse event delivery | Numerical reference |
| Apple MPS | float32, custom Metal CSR gather | Experimental; trajectories can differ |
| Apple MPS optimized | float16, active-row Metal gather with four marking lanes | Explicit approximation; firing differs from FP32 |

Select **Data & model → Brain execution → Use Apple GPU**. The footer reports the actual neural tensor device and precision. Switching pauses the animal, saves the original state in `data/execution-backups/`, transfers every neural state tensor and pending delayed spike, and saves the migrated state. Simulation time is preserved. **Run** resumes; **Step** advances one command cycle. GPU mode persists across restart through `data/live-state.pt`.

The [60 Hz scheduling mode](SCHEDULING.md) uses 960 Hz neural updates, 16.667 ms command cycles, paced result publication, smooth display interpolation and a performance indicator. In this mode, **Step** advances one 16.667 ms cycle; older 50 Hz checkpoints retain 20 ms cycles. The selected FP16 or FP32 precision persists through clock changes and resets.

Switching to higher precision cannot recover rounding already introduced by FP16 or FP32. The backup retains the complete state from before conversion. To restore it, stop the server normally, preserve the current live checkpoint, copy the chosen backup to `data/live-state.pt`, and restart. Checkpoints record device, precision and kernel version; incompatible or unavailable execution modes fail explicitly.

## Implementation

The installed PyTorch 2.14.0 already supports this Mac's GPU. No CUDA package is needed. MPS cannot represent our float64 sparse CSR tensor directly. `metal_dynamics.py` uses PyTorch's [documented Metal shader compilation API](https://docs.pytorch.org/docs/stable/generated/torch.mps.compile_shader.html) with ordinary GPU tensors for CSR offsets, indices and weights.

Each neural tick (0.1 ms by default; explicit 1 ms approximation available below) dispatches three ordered kernels: neuronal integration and threshold detection; delayed synaptic gathering; input delivery and resets. Each postsynaptic row uses one 32-thread SIMD group with fixed reduction ordering and compensated partial sums. There are no floating-point atomic writes. All 166,700 imported neurons and 25,582,938 directed pairs remain present. Integer source synapse counts remain unchanged; GPU indices are range-checked int32. GPU efficacy and state use the selected float32 or float16 precision. The optimized FP16 engine adds integer destination marking before gathering, retaining every neural update.

Brain state stays on GPU between ticks. External sensory input transfers to GPU per 50 or 60 Hz body interval; motor readout transfers back to CPU. MuJoCo, sensors, anatomical inventory queries, and the isolated paper experiment still use CPU. Gesture adapter training now has a GPU FP16 forward / FP32 backward path; see `GESTURE_LEARNING.md`. Other training workers retain their existing execution paths. Poisson experiments draw CPU float64 uniforms with the reference RNG and transfer stimulus events to GPU. No unsupported-operation CPU fallback is enabled for neural kernels.

## Validation and measured limits

On the M1 Pro, all 70 tests passed, including real GPU execution. The small Brian2 comparison covers recurrent excitation/inhibition, strict threshold, refractory freezing, resets and both zero and 1.8 ms delays. GPU spike sequences match exactly in those fixtures; state error is below 0.00005 mV. GPU tests also cover CSR direction/high-degree rows, Poisson batching, soma clamping, device migration, API status and exact same-backend brain/body checkpoint continuation.

A separate complete-graph check starts both devices at rest and injects identical constant drive for 40 ms (400 ticks). The [recorded result](results/mps-validation.json) reports:

| Measurement | CPU float64 | MPS float32 |
|---|---:|---:|
| Neural compute time | 0.892 s | 1.988 s |
| Total spikes | 28,997 | 28,985 |

Spike counts differ in 28 neurons, by one spike each. Maximum final voltage difference is 6.997 mV; maximum current-state difference is 522.752 mV. Near-threshold rounding can change a spike/reset and propagate through recurrent connections. This is a short endpoint comparison, not a full spike-train comparison, a biological fidelity score or long-run equivalence evidence. GPU execution is currently slower on this workload. Measurements exclude body mechanics and are not sustained performance guarantees.

The [live kitchen check](results/mps-live-step.json) advanced the saved animal from 0.20 to 0.22 seconds on MPS in 1.073 seconds, including sensing and body mechanics (0.966 seconds neural computation). The app then restored the original 0.20-second state, paused on MPS; the physical snapshot was unchanged. The footer reports lifetime timing, which can include earlier CPU intervals after a device switch.

Reproduce from the repository root on an MPS-capable host:

```bash
PYTHONPATH=backend uv run pytest -q
PYTHONPATH=backend uv run python scripts/validate_gpu.py
```

Hardware-specific tests skip when MPS is unavailable; a skipped test is not GPU validation. The HTTP API exposes `GET /api/execution` and `POST /api/execution` with `{"device":"mps"}` or `{"device":"cpu"}`. Python callers can use `FullBrain(path, device="mps")` or migrate a paper-profile brain with `brain.set_device("mps")`.


## Offline mixed-precision experiment

`MetalDynamics(weights, weight_dtype=torch.float16)` stores synaptic efficacies as FP16 after FP32 conversion. Each efficacy is explicitly converted back to FP32 before multiplication; compensated partial sums, reductions, voltages, currents, delayed signals and rates remain FP32. CSR offsets/indices remain int32. Every imported edge and neuron is retained. Conversion fails if a nonzero efficacy would become zero, or if a weight becomes nonfinite. This does not prevent smaller rounding errors or changes to spike timing.

This is an isolated benchmark option, not a live execution selector. Its kernel identifier is `metal-csr-w16-state32-v1`; ordinary FP32 remains `metal-csr-f32-v1`. Execution summaries report weight precision separately from state precision. Live checkpoint loading rejects the experimental kernel instead of silently resuming it with different weights. Do not use this offline option to save a live animal checkpoint.

```bash
PYTHONPATH=backend uv run python scripts/benchmark_mixed_precision.py --duration-ms 100 --trials 3
```

The script loads a separate full brain from the imported graph, never reads or writes live checkpoints, and writes `docs/results/mixed-precision.json`. Both GPU modes start from reset and receive identical drive. It warms both modes before alternating their timing order; load and shader compilation are excluded. A separate validation pass compares endpoints and per-neuron spike counts in 1 ms bins. These bins cannot resolve within-bin timing differences. The CPU FP64 endpoint is included as a numerical reference, not biological ground truth. Results cover one short stimulus without body dynamics and may vary with concurrent GPU load.


### Recorded mixed-precision result

On this M1 Pro, three alternating timed runs of 100 ms (1,000 neural ticks) gave the following [result](results/mixed-precision.json):

| Measurement | FP32 weights | FP16 weights, FP32 arithmetic/state |
|---|---:|---:|
| Median wall time | 4.372 s | 4.241 s |
| Weight buffer | 102.33 MB | 51.17 MB |
| CSR buffers including indices | 205.33 MB | 154.16 MB |
| Total spikes | 103,411 | 103,693 |

Mixed mode achieved 1.031× throughput, about a 3% improvement in this short test. Weight storage halved; total CSR storage fell about 25%, not total process memory. Retained source counts and connectivity were identical.

Numerical differences were material: 10,429 neurons had at least one different 1 ms spike-count bin, first appearing in the 25–26 ms bin. Final cumulative spike counts differed in 2,836 neurons, by up to four spikes. These are differences from FP32, not a biological accuracy score. The CPU FP64 reference also differs from both GPU modes; its single timed run took 2.275 s, outside the repeated GPU timing comparison.

Mixed FP16-weight/FP32-state execution remains benchmark-only; the live FP16 option uses the optimized all-half kernel described below. The mixed kernel remains available for reproducible offline experiments; no live simulation or checkpoint was migrated.

Validation after this change: 109 backend tests passed with MPS available, including the new FP16 gather, overflow/underflow rejection, unchanged-state-precision and exact-representable-weight scheduling checks.


## Offline all-FP16 and FP8 checks

Run `PYTHONPATH=backend uv run python scripts/benchmark_mixed_precision.py --all-fp16` to add an all-FP16 candidate to the repeated comparison. It writes `docs/results/all-fp16.json`, preserving the earlier mixed-only report. Kernel `metal-csr-f16-v1` uses half-precision weights, neuron state, delayed signals, inputs, coefficients, products and sums, including compensated summation. Shader literals and intermediate variables are half-precision as well. Host-computed coefficients are rounded to FP16 before dispatch. Source graph data remains unchanged; neuron IDs, connectivity indices, refractory ticks, spike counters and timestep indices retain exact integer types. The mode is offline-only; live checkpoint loading rejects its distinct kernel/precision.

`PYTHONPATH=backend uv run python scripts/probe_gpu_precision.py` probes allocation and multiplication on temporary MPS tensors. On the installed PyTorch 2.14.0 stack, FP16 multiplication works. Both `float8_e4m3fn` and `float8_e5m2` allocate but fail multiplication with an undefined-type error, recorded in `docs/results/gpu-low-precision-support.json`. Allocation alone does not establish arithmetic support. No all-FP8 simulation benchmark is claimed; storing bytes and decoding to a wider format would be a different experiment.


### Recorded all-FP16 result

The [100 ms full-graph comparison](results/all-fp16.json), with three warmed trials in rotating order, measured:

| Mode | Median wall time | Total spikes |
|---|---:|---:|
| FP32 | 4.359 s | 103,411 |
| FP16 weights / FP32 state and arithmetic | 4.207 s | 103,693 |
| FP16 weights, state and arithmetic | 3.901 s | 82,284 |

All-FP16 achieved 1.117× throughput (10.5% less wall time) versus FP32. It produced 20.4% fewer spikes in this stimulus, with different final or intermediate behavior despite identical connectivity. 18,381 neurons had at least one different 1 ms spike-count bin, first appearing at 12–13 ms. All endpoint state tensors were finite, and integer tick counts remained 1,000. Finiteness does not establish physiological accuracy. Lower activity also changes the integration workload, so this is observed end-to-end simulation performance, not an isolated arithmetic throughput measurement.

Entering MPS without an explicit precision uses FP32. The optimized all-FP16 engine can be selected explicitly in the live app; the baseline all-half kernel in this benchmark remains offline. All-FP8 is unavailable for the installed PyTorch/MPS arithmetic path, rather than silently emulated at higher precision.

All-FP16 validation: 111 backend tests passed with real MPS execution. Added checks cover half-precision state and inputs, exact integer timing, incompatible-buffer rejection, recurrent signal delivery, delays, clamping, reset and timestep batching. These are implementation checks, not biological validation.


## FP16 active-row optimization

`ActiveRowMetalDynamics` builds an additional outgoing connectivity index to identify destinations of delayed nonzero signals. Integration clears integer destination flags; a separate dispatch marks them using idempotent integer atomic stores. Only marked destinations scan their incoming CSR rows. Every neuron still integrates, checks its threshold and completes the reset/rate update on every 0.1 ms tick. The original one-lane marker is `metal-active-rows-f16-v1`; the class defaults to four lanes (`metal-active-rows-f16-v1-mark4`), measured below. Historical benchmark commands explicitly retain the one-lane implementation.

Marked rows retain exactly the baseline FP16 connection order, 32-lane assignment, compensated partial sums and reduction. Zero terms inside a marked row are retained because skipping them can change compensated floating-point sums. No floating-point atomics, precision changes, connection pruning or timesteps skipped are introduced by this optimization. Scratch flags are rebuilt each tick. The additional index and flags increase these GPU connectivity buffers from 154.16 MB to 257.83 MB. The extra marking pass can reduce benefits at high activity.

Reproduce the phase replay and full-graph comparisons with:

```bash
PYTHONPATH=backend uv run python scripts/profile_fp16.py
PYTHONPATH=backend uv run python scripts/benchmark_fp16_optimization.py
PYTHONPATH=backend uv run python scripts/benchmark_fp16_optimization.py --duration-ms 1000 --trials 2 --output docs/results/fp16-optimization-1s.json
```

The [phase replay](results/fp16-profile.json) measured about 2.7 ms per baseline connection scan, even with no signals, versus approximately 0.11 ms combined integration/finalization. These are synchronized repeated-dispatch wall times on synthetic buffers, not Instruments hardware-counter measurements or an additive simulation profile. MPS event timing stalled; the isolated process was stopped and the recorded profile uses explicit synchronization instead.

The [100 ms comparison](results/fp16-optimization.json), with three warmed alternating runs, measured median wall times of 2.819 s (baseline FP16) and 0.703 s (active rows): 4.01× throughput. Both emitted 82,284 spikes. All nine neural tensors matched exactly at the endpoint; byte-level hashes of every neural tensor, pending delays, random state and tick index matched at all 100 sampled millisecond boundaries. Small tests additionally compare every tick, zero and nonzero delays, repeated flag clearing, dense and sparse signals, clamping and Poisson batching. This establishes tested numerical agreement with FP16, not fidelity to FP32 or biology.

The four-lane variant is available in the live app. Benchmark scripts use isolated brains and never migrate the live animal or alter its saved state. Times vary with GPU load, so speedups compare modes in the same run rather than against earlier reports.


A [one-second extension](results/fp16-optimization-1s.json), with two warmed alternating trials and 10,000 ticks per trial, measured 28.446 s for baseline FP16 versus 9.235 s for active rows: 3.08× throughput. Both produced 1,024,513 spikes. All nine neural tensors matched at the endpoint, and the full-state byte hashes matched at every one of the 1,000 sampled millisecond boundaries. The sampled active-row fraction averaged 8.45% (maximum 24.67%). This remains slower than real time and is one fixed-stimulus validation, not a general performance or biological-equivalence guarantee.

### Cooperative destination marking

Four GPU threads now share each source neuron's outgoing destinations, distributing work from neurons with many connections. They only set integer flags; incoming FP16 sums retain their original order. Connectivity storage remains 257.83 MB, with all 166,700 neurons, 25,582,938 directed pairs and 0.1 ms timesteps preserved.

The [full-graph comparison](results/fp16-mark4-1s.json) used three warmed, alternating trials of one simulated second:

| Marker | Median wall time | Total spikes |
|---|---:|---:|
| One thread per source | 12.317 s | 1,024,513 |
| Four threads per source | 8.968 s | 1,024,513 |

This gives 1.373× throughput, or 27.2% less wall time, against the baseline measured in the same run. All nine endpoint tensors and all 1,000 sampled full-state hashes matched exactly. It establishes agreement with the earlier FP16 implementation for this stimulus; it does not remove FP16's previously measured divergence from FP32. The four-lane engine is also the explicit live FP16 option.

The [marker microbenchmark](results/fp16-marking-profile.json) compares 1, 4, 8 and 32 threads per source on fixed sparse and dense signal buffers. Four performed well across densities; higher counts sometimes won at intermediate activity. These synchronized clear-and-mark timings include submission overhead and do not predict whole-simulation speed by themselves.

```bash
PYTHONPATH=backend uv run python scripts/profile_fp16_marking.py
PYTHONPATH=backend uv run python scripts/benchmark_fp16_optimization.py --mark-lanes 4 --duration-ms 1000 --trials 3 --output docs/results/fp16-mark4-1s.json
```

### Fused dispatch experiment

`FusedActiveRowMetalDynamics` combines gathering and finalization, reducing four dispatches per tick to three. It snapshots delayed signals before clearing the consumed queue to avoid concurrent readers losing input, including with zero delay. The [100 ms comparison](results/fp16-fused.json) preserved sampled states exactly but increased median runtime from 0.981 s to 1.100 s. It remains an explicit offline experiment, not the default. Reproduce with `scripts/benchmark_fp16_optimization.py --fused` using the same Python invocation above.

Validation: 131 backend tests passed with MPS available. Candidate checks cover the four-thread default, explicit 1/8/32-thread variants and fusion, including exact delivered sums, stale-flag clearing, every-tick states, zero/nonzero delays, clamping and Poisson batching.

## Explicit coarse neural clock

**Data & model → Brain execution → 1 ms · faster approximation** reduces neural updates from 10,000 to 1,000 per simulated second. It changes temporal resolution; it is not an equivalent optimization of the 0.1 ms reference. All neurons and edges remain. Body integration and the 20 ms brain/body exchange interval remain unchanged. No automatic timestep adaptation or dropped steps is used.

The API accepts `POST /api/execution` with `{"device":"mps","neural_dt_ms":1}`. A device-only request preserves the selected interval. The 0.1 ms option restores reference clock resolution, but cannot undo a trajectory already altered by coarse timing. Execution summaries expose the interval and effective delay/refractory durations. This clock option preserves the selected precision.

`Physiology.timing_rounding="ceil"` explicitly rounds the nominal 1.8 ms delay to 2 ms and nominal 2.2 ms refractory duration to 3 ms. The nominal parameters are retained in checkpoints. Switching pauses and backs up the animal; continuous neural state, counts, RNG, body state and simulation time survive. Pending signal delivery and refractory release move to the next new clock boundary, never earlier. Events sharing a boundary are summed. Refining the clock retains any longer pending queue needed by previously scheduled signals. A failed migration/save restores the previous execution state.

The [FP32 benchmark](results/timestep-float32.json) compares three alternating trials of **200 ms simulated time**: median wall time falls from 8.888 s to 0.892 s (9.96× throughput). Normalized to one simulated second, these are 44.44 and 4.46 wall seconds; that normalization is not a separately measured one-second trial. Total spikes change from 232,610 to 224,572, with differing counts in 11,023 neurons. This is one fixed input from reset, without body, sensing or UI cost.

```bash
PYTHONPATH=backend uv run python scripts/benchmark_timestep.py
PYTHONPATH=backend uv run python scripts/benchmark_timestep.py --precision float16 --duration-ms 1000 --trials 2
```

The browser receives complete poses after a 20 ms brain/body batch finishes. At slow neural throughput that can look like a pose jump about once per wall second. Coarser timing shortens that batch; it does not add interpolated poses or guarantee real-time motion.

The [optimized FP16 comparison](results/timestep-float16.json), with two alternating trials of one simulated second, measured 9.209 s at 0.1 ms and 2.309 s at 1 ms (3.99× throughput). Total spikes increased from 1,024,513 to 1,205,687, with different counts in 19,446 neurons. Different activity levels change the active-row kernel's workload, so ten times fewer ticks does not necessarily give a tenfold speedup. This remains a separate offline FP16 experiment.

The live FP32 smoke check at 1 ms completed four successive 20 ms brain/body batches in 0.161–0.186 wall seconds each (0.090–0.102 s neural compute). The requests, including response snapshots, took 0.196–0.279 s. These short measurements cover startup from the saved state, not sustained frame delivery. The original time, body pose and spike count were restored afterward; the checked state was paused at 1 ms. The UI and saved checkpoint both report the selected mode.

The resulting simulated rates are 50 Hz sensory/motor exchange, 1,000 Hz brain updates and 10,000 Hz physical integration: 20 neural ticks and 200 physics steps per exchange. Current amplitudes remain unchanged because the analytic integration coefficients already account for elapsed time. Firing-rate filters also use the configured timestep. A universal amplitude multiplier would double-count duration for those inputs; coarser sampling still cannot preserve all threshold crossings or feedback.

Validation after clock integration: 139 backend tests passed with MPS execution; the production frontend build passed. Desktop (1440×1100) and mobile (390×844) Chrome checks covered the clock controls, selected state, error-free page, saved-state metadata and absence of horizontal overflow. Longer runs, other browsers and a 120/960 Hz exchange schedule have not been validated.


## Live optimized FP16

Select **Data & model → Brain execution → Use Apple GPU · optimized float16**, or post `{"device":"mps","precision":"float16"}` to `/api/execution`. The selected kernel is `metal-active-rows-f16-v1-mark4`. Device-only and clock-only requests preserve the current GPU precision. Explicit `float32` returns to the original GPU kernel; CPU requires `float64`.

Conversion preserves simulation/body clocks, pending signal slots, RNG, neuron IDs and integer counters. Floating neural buffers are rounded to FP16; overflow rejects the conversion before changing the engine. The original state is backed up before migration. Complete live checkpoints record and validate the precision and kernel, including exact continuation after restoration. Reset, scene changes, physiology changes and paper-profile physical-policy loading retain the selected precision. The legacy normalized controller remains CPU float32. MuJoCo and sensory calculations remain on CPU in their existing numerical formats.

Run `PYTHONPATH=backend uv run python scripts/benchmark_live_precision.py` to compare copies of the same saved animal in alternating FP32/FP16 trials. Each trial advances 60 complete cycles, including sensing, brain, body and snapshot preparation; loading, one warm-up cycle, HTTP transport and browser rendering are excluded. This is separate from the brain-only benchmarks above. FP16 trajectories can diverge, so different activity/contact workloads contribute to the measured result. No biological equivalence or real-time speed is implied.

The [complete-cycle comparison](results/live-fp16.json) ran two alternating trials per precision on the M1 Pro, at 60 command Hz / 960 neural Hz, each covering one simulated second after warm-up:

| Measurement | FP32 | Optimized FP16 |
|---|---:|---:|
| Median wall seconds per simulated second | 12.73 | 10.53 |
| Mean cycle duration (ms), median trial | 212.2 | 175.5 |
| Neural wall seconds per trial | 4.54 | 2.81 |

This workload showed 1.21× throughput (17.3% less wall time). The complete backend remains slower than real time. These controlled results exclude HTTP and browser load; the live performance indicator can therefore differ. No neurons, connections or scheduled neural ticks were removed.


Validation: 151 backend tests passed with MPS available; the production frontend build passed. Browser testing at `http://127.0.0.1:8000/` exercised the FP16 selector, page reload, Run and Pause without console errors. The full-size live checkpoint restored the same 1.20-second time and physical snapshot, retaining 166,700 neurons, 25,582,938 connections, FP16 and the 60/960 Hz clocks. That short rendered session reported about 5.0 cycles per wall second (last cycle 206 ms); it is a separate observation from the controlled backend comparison. The animal was saved and left paused. FP16 biological fidelity, longer trajectories and other hardware remain unvalidated.

## Peripheral GPU matrix trials

The [20-cycle CPU profile](results/peripheral-profile.json) includes full FP16 brain/body steps and snapshots. Body physics accounts for about 61% of measured time, brain integration 27%, sensing 7%, and muscle readout 2%. GPU synchronization is included in brain time; these shares are workload-specific.

`MatrixMotorReadout` represents every existing mapped motor transmission as a compact dense matrix. It gathers mapped motor rates on MPS and calculates all channel means with `matrix @ rates`, returning only the actuator vector to CPU. This introduces FP32 readout arithmetic in place of CPU double weighted means; neural state precision and graph topology remain unchanged. It is an isolated experimental implementation, not the live readout default.

| Experiment | Current implementation | Candidate | Outcome |
|---|---:|---:|---|
| Muscle readout, median per call | CPU 2.233 ms | GPU matrix 0.710 ms | 3.15× faster locally |
| Complete 30-cycle trial (0.5 simulated seconds) | CPU readout 5.674 s | GPU readout 5.822 s | No complete-cycle gain |
| Two-eye quadrature/receptor processing | CPU 0.142 ms | GPU matrix 1.387 ms | Slower including required transfers |
| Body-only 60-cycle trial | Python dispatch 8.582 s | Native `nstep` 8.674 s | No demonstrated gain |

The [muscle trials](results/matrix-readout.json) alternate order, using five 100-call samples per mode and two full-cycle trials per mode from the same saved state. The synthetic half-rate sweep produced equal final float32 actions; this is not proof of equality for arbitrary rates. [Eye processing](results/visual-matrix.json) used seeded RGB input for all 1,709 facets with seven rays each; maximum receptor difference was 1.16e-7. Ray intersections remain CPU MuJoCo work. [Native batching](results/physics-batch.json) retained all substeps and final fractional steps and produced identical complete physical endpoints in all eight trials. Small wall-time differences are not sustained speed guarantees. These candidates have not replaced the live implementations.

Reproduce with `PYTHONPATH=backend uv run python scripts/benchmark_matrix_readout.py`, `scripts/benchmark_visual_matrix.py`, or `scripts/benchmark_physics_batch.py`. Routing tests (`tests/test_matrix_readout.py`) pass on CPU and MPS, covering weighted means, shared inputs, unmapped channels, changing gains and clipping.

### Actual GPU physics compatibility

[MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) provides a separate GPU physics implementation. The official [Apple JAX plug-in](https://developer.apple.com/metal/jax/) is experimental. These probes install into a temporary environment, leaving FlyLab dependencies and the saved animal unchanged:

```bash
uv venv /tmp/flylab-mjx-probe --python .venv/bin/python
uv pip install --python /tmp/flylab-mjx-probe/bin/python 'mujoco-mjx==3.13.0' 'jax-metal==0.1.1' 'jax==0.11.1' 'jaxlib==0.11.1'
/tmp/flylab-mjx-probe/bin/python scripts/probe_mjx_metal.py
uv pip install --python /tmp/flylab-mjx-probe/bin/python 'jax==0.4.34' 'jaxlib==0.4.34'
/tmp/flylab-mjx-probe/bin/python scripts/probe_mjx_metal.py --output docs/results/mjx-metal-compatible-probe.json
PYTHONPATH=backend uv run python scripts/probe_mjx_body.py --python /tmp/flylab-mjx-probe/bin/python
```

The [current JAX trial](results/mjx-metal-probe.json) detected the M1 Pro but failed compilation with an unsupported StableHLO attribute. The [older compatibility trial](results/mjx-metal-compatible-probe.json) successfully executed a 32×32 matrix product on `METAL:0` and imported MJX. The [full-body probe](results/mjx-body-probe.json) then rejected the unchanged 103-DoF, 186-actuator, 102-tendon body during model conversion: `PLANE, MESH margin/gap not implemented`. No GPU physics step ran, so no GPU physics speedup is claimed. Collision margins were not removed to force compatibility. Supporting these contact semantics is a prerequisite to a faithful full-body MJX comparison on this stack.

## Concurrent GPU brain and CPU body

The live workbench supports **Overlap brain & body** with a one-command buffer; see [scheduling](SCHEDULING.md). Physics consumes the previous excitation while the FP16 brain calculates the next one. Both finish each simulated interval before publishing. This preserves all neurons, connections, neural steps and physical substeps, but explicitly adds 16.667 simulated ms of command delay at 60 Hz. It is not a measured biological transmission delay.

The [complete-cycle comparison](results/coupling.json) uses the M1 Pro, FP16, 60/960 Hz clocks, the full 166,700-neuron / 25,582,938-pair graph, and one CPU tensor thread as configured in the server. Each of two alternating trials covers 60 cycles (one simulated second), following one untimed warm-up cycle from the same saved state. Loading, transport and rendering are excluded.

| Mode | Median wall seconds per simulated second | Mean cycle ms, median trial |
|---|---:|---:|
| Original serial coupling | 10.34 | 172.4 |
| Sequential execution with the same one-cycle delay | 10.22 | 170.4 |
| Concurrent execution with one-cycle delay | 7.53 | 125.6 |

Concurrent execution gives **1.36× throughput** versus the matching delayed reference (26.3% less wall time), and 1.37× versus original serial coupling. The delayed serial and concurrent endpoints match exactly across all four trials: neural tensors, RNG state, complete MuJoCo integration state and pending muscle command. Original serial coupling follows a different trajectory because its timing differs. This is a finite numerical check, not biological validation or a guarantee for other scenes and activity levels. The simulation still runs in slow motion (about 8 backend cycles per wall second).

Reproduce with `PYTHONPATH=backend .venv/bin/python scripts/benchmark_coupling.py`. The script operates on independent copies and never writes the input checkpoint. Result JSON records graph/checkpoint hashes, engine versions, timing, source time and endpoint hashes.

Validation: **167 backend tests passed** with MPS available, and the frontend production build passed. The browser exercised overlap selection, Run, Pause, Save and Resume without console errors. A 91-cycle rendered run ended at 1.516667 simulated seconds; the final live indicator read 6.1 cycles per wall second and 166 ms for the latest cycle. Rendering and variable activity make this different from the controlled benchmark. The [live restoration check](results/coupling-live.json) verified all saved neural tensors, RNG, complete physics state, body tensors and the buffered command exactly after restoration. The server remains up with FP16, 60/960 Hz clocks and overlap enabled; the animal is saved and paused.
