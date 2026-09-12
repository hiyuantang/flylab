# GPU execution

The live MaleCNS brain supports two explicit execution modes:

| Mode | Neural arithmetic | Status |
|---|---|---|
| CPU | float64, sparse event delivery | Numerical reference |
| Apple MPS | float32, custom Metal CSR gather | Experimental; trajectories can differ |

Select **Data & model → Brain execution → Use Apple GPU**. The footer reports the actual neural tensor device and precision. Switching pauses the animal, saves the original state in `data/execution-backups/`, transfers every neural state tensor and pending delayed spike, and saves the migrated state. Simulation time is preserved. **Run** resumes; **Step** advances 20 ms. GPU mode persists across restart through `data/live-state.pt`.

Switching back to CPU cannot recover rounding already introduced by float32. The original backup retains that prior float64 state. To restore it, stop the server normally, preserve the current live checkpoint, copy the chosen backup to `data/live-state.pt`, and restart. Checkpoints record device, precision and kernel version; incompatible or unavailable execution modes fail explicitly.

## Implementation

The installed PyTorch 2.14.0 already supports this Mac's GPU. No CUDA package is needed. MPS cannot represent our float64 sparse CSR tensor directly. `metal_dynamics.py` uses PyTorch's [documented Metal shader compilation API](https://docs.pytorch.org/docs/stable/generated/torch.mps.compile_shader.html) with ordinary GPU tensors for CSR offsets, indices and weights.

Each 0.1 ms tick dispatches three ordered kernels: neuronal integration and threshold detection; delayed synaptic gathering; input delivery and resets. Each postsynaptic row uses one 32-thread SIMD group with fixed reduction ordering and compensated partial sums. There are no floating-point atomic writes. All 166,700 imported neurons and 25,582,938 directed pairs remain present. Integer source synapse counts remain unchanged; GPU indices are range-checked int32. GPU efficacy and state use float32.

Brain state stays on GPU between ticks. External sensory input transfers to GPU per 20 ms body interval; motor readout transfers back to CPU. MuJoCo, sensors, anatomical inventory queries, training workers and the isolated paper experiment still use CPU. Poisson experiments draw CPU float64 uniforms with the reference RNG and transfer stimulus events to GPU. No unsupported-operation CPU fallback is enabled for neural kernels.

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
