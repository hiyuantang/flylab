# Brain fidelity and continuation

FlyLab preserves every neuron and connection in its imported identified-neuron graph. Execution speed never changes population size, integration time, or transmitter signs. Numerical precision changes only through explicit execution-mode selection. A slow calculation advances less simulated time per wall-clock second. No neural or physical steps are dropped to catch up.

See the [dataset architecture audit](DATASET_ARCHITECTURE_AUDIT.md) for neuron/type inventories, measured feedback wiring, selection-boundary analysis, and separate structural fidelity targets.

## Two specimens, two experiments

The embodied workbench uses **MaleCNS v1.0**: 166,700 annotated neurons, 25,582,938 unique directed pairs, and 124,177,617 retained synapses. The importer selects annotations with a non-null superclass. Its 44,877 unclassified annotation rows and all crossing connections remain outside this identified-neuron graph and are reported in the manifest. This boundary is unresolved data interpretation, not a performance optimization. Importing the graph does not establish biological completeness.

The isolated paper reference uses **FlyWire female v630**, exactly as supplied by [Shiu et al.](https://www.nature.com/articles/s41586-024-07763-9): 127,400 neurons, 14,687,178 connection rows, and 52,793,639 synapses. Source tables are SHA256-checked against commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` of the [authors' repository](https://github.com/philshiu/Drosophila_brain_model). Original signed counts and neuron ordering are preserved. FlyWire IDs are never interpreted as MaleCNS IDs.

## Verified neuronal equations

The `shiu-2024-linear-delay-v1` profile uses mV and ms:

- Rest/reset: −52 mV; threshold: strictly greater than −45 mV.
- Membrane time constant: 20 ms; synaptic time constant: 5 ms.
- Refractory period: 2.2 ms; transmission delay: 1.8 ms; integration step: 0.1 ms.
- Efficacy: signed anatomical count × 0.275 mV, without postsynaptic normalization.
- Exact linear integration of `dv/dt = (−52 − v + g)/20` and `dg/dt = −g/5` between events.
- State integration → threshold → delayed synapses/input → reset, matching Brian2 scheduling. Both state variables freeze while refractory; writes into refractory variables are discarded. Spiking resets voltage and synaptic current.

Float64 tests compare every voltage, synaptic state, and spike time against Brian2 2.10.1, including inhibitory/recurrent connections, refractory input, zero delay, and 1.8 ms delay. Tested state error is below 2 × 10⁻¹¹ mV and spike sequences match exactly. These are numerical verification tests, not evidence that one LIF parameter set captures real cell-type physiology.

All neurons integrate every tick. Sparse event delivery visits outgoing edges of neurons that actually emitted a spike; multiplying other edges by zero is unnecessary. This changes execution, not network topology. Anatomical counts remain int64. The CPU reference state and synaptic efficacy calculations use float64. The optional MPS mode uses float32 and a complete CSR gather, with numerical differences disclosed separately.

## Sensory and physical assumptions

The paper experiment delivers Bernoulli-discretized Poisson voltage input at the paper's time step: a 68.75 mV jump with probability `rate_Hz × 0.0001`. Target receptors have zero refractory duration, following the source. No-input, 100 Hz sugar, and 100 Hz sugar + 100 Hz bitter conditions use the published receptor ID lists. The RNG is reproducible, but differs from Brian2's generator; equal seeds do not imply identical stochastic input streams across simulators or conditions. No parameters are trained.

MaleCNS transfers the paper equations but uses its own annotations. ACh is excitatory; GABA, histamine, and default glutamate are inhibitory. Unresolved/modulatory effects default to zero fast synaptic current because receptor-specific action is unknown. These neurons, states, and anatomical edges remain present. This differs from the paper's published signed-count convention and is disclosed as an assumption.

Workbench sensory controls supply assumed continuous currents in threshold-relative units, converted using the 7 mV threshold gap. This differs from the paper's Poisson protocol. Visual receptor tuning, cross-specimen eye registration, same-side muscle assignments and firing-rate-to-force conversion are unvalidated. Compound vision supplies individual facet inputs through inferred columns; legacy vision retains side pooling. See [sensory evidence and limits](SENSING.md). Every 20 simulated ms, sensed body state drives the brain, filtered motor rates excite the supported actuators, and MuJoCo integrates 20 ms of mechanics. This coupling interval is a modeling approximation; it is not adaptive and never increases under load. The experiment has no gait oscillator or posture controller. Unsupported actuators receive zero excitation, while their neurons continue to run. The expanded mapping with pretarsal and peripheral mechanics covers 454 of all 815 brain/VNC motor neurons, with 361 unresolved outputs; of the connected identities, 436 are supported, 14 tentative, and four muscle-family-only; see the [versioned neuromuscular inventory](NEUROMUSCULAR_MAPPING.md). This coverage does not measure biological fidelity.

## Continuation and time

**Save state** atomically writes `data/live-state.pt`. It includes every neuron's voltage, synaptic current, rate, refractory history, spike counts, output gain, pending delayed events, RNG, interventions, sensory settings, scene version, full MuJoCo integration state and muscle activation. **Resume saved** validates graph, compiled body, dynamics, sensor and MuJoCo versions before replacing the workbench; it restores paused. Tests verify exact neural and physical continuation against an uninterrupted run.

Normal server shutdown saves the last completed state; startup resumes it paused. Force-killing the process can lose work since the last save. An incompatible checkpoint produces an explicit error rather than a new or smaller brain. Reset and scene/sensory reconfiguration explicitly start a new episode.

The footer separates simulated time from computation time and reports their ratio. Idle/paused time is excluded from computation time. Pacing controls cap execution rate; they never promise real-time performance. The 3D brain view uses measured `somaLocation` coordinates for 139,662 imported neurons. The 27,038 without soma coordinates remain searchable; no locations are fabricated. A shared axis transform and uniform scale preserve relative positions. Brain/VNC display filters use superclass annotations, including ascending classes with VNC, and do not define neuropil boundaries. Colors encode each neuron’s simulated filtered firing rate. The view requests fresh activity after completed simulation cycles, coalescing updates into one request at a time and remaining idle for unchanged or hidden views. A 100 ms GPU color fade smooths display transitions without extrapolating activity or changing the simulation. Connection details refresh at most once per second. Incoming/outgoing links use exact imported neuron pairs and integer synapse counts; straight lines connect soma locations and are not anatomical axon trajectories. Only the current 100-partner page is drawn, while all partners are paginated. This is a display limit only: no simulated neurons or edges are removed.

The spatial layout is transferred once as binary arrays (3.50 MB); activity uses a separate float32 display payload (0.667 MB) only when the simulation changes. The neural precision is unchanged. A single batched point cloud and batched lines render on demand; hidden tabs skip activity polling. Adjacency results use a bounded 32-neuron cache. Selecting a new neuron scans the original presynaptic index once to find outgoing partners; subsequent state refreshes reuse its adjacency.

## Reference result and acceleration status

On the local Apple M1 Pro, CPU float64, seed 42, one 1-second trial per condition produced:

| Condition | MN9 output 1 (Hz) | MN9 output 2 (Hz) | Total spikes | Compute seconds |
|---|---:|---:|---:|---:|
| No input | 0 | 0 | 0 | 13.30 |
| Sugar, 100 Hz | 73 | 58 | 10,214 | 16.04 |
| Sugar + bitter, each 100 Hz | 6 | 11 | 9,146 | 16.51 |

This reproduces the qualitative suppression expected in this protocol. It is a **single-seed smoke test**, not reproduction of the paper's 30-trial statistics or 164 predictions. The paper's reported agreement is not a percentage of all fly behaviors. MN9 firing is not validated feeding muscle motion. The result is recorded in `docs/results/shiu-feeding-42-1000ms.json`.

The first embodied MaleCNS laboratory step computed 20 ms in 0.477 s wall time, with 10,486 spikes and one excited muscle channel. This short measurement is not a sustained performance or locomotion benchmark.

PyTorch 2.14.0 detects the M1 Pro GPU. Experimental MPS execution now uses custom Metal kernels over the complete CSR graph because MPS rejects native float64 and sparse CSR tensors. CPU float64 remains the reference. GPU float32 passes small Brian2 scheduling tests but differs on the full-network stimulus; it is not validated as an equivalent replacement. See [GPU measurements, controls and limitations](GPU_EXECUTION.md). Performance claims require end-to-end measurement, including mechanics and sensory coupling.

## Reproduce

From the repository root:

```bash
uv sync --python 3.12
PYTHONPATH=backend uv run pytest tests/test_paper_dynamics.py tests/test_live_state.py -q
PYTHONPATH=backend uv run python scripts/reproduce_paper.py --download --duration-ms 1000 --seed 42
```

The download is approximately 90 MB. Paper data and live checkpoints stay in ignored `data/`. **Data & model → Run full-graph reference** runs the same isolated protocol after the source files are installed. The original reference circuit and posture utilities remain for historical tests; the web controller does not offer them or silently fall back to them.

## Manual dopamine teaching

The optional [dopamine teaching model](DOPAMINE.md) maps signed user feedback to
annotated PAM01/PPL101 stimulation and local depression of existing KCγ→MBON
connections. It preserves the measured topology and stores learned factors
separately. Its physiological parameters, compartment transfer, and plasticity
law are explicit assumptions; it is not yet validated behavioral conditioning.

## Whole-population evidence

The [neuron evidence inventory](NEURON_EVIDENCE.md) documents every imported neuron with source annotations, transmitter prediction scores, anatomical wiring provenance, explicit physiology assumptions and implemented peripheral interfaces. Data & model separates these into searchable, paginated pages. Dataset support describes provenance; it is not an overall confidence percentage or functional validation.
