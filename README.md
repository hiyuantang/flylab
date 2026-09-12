# FlyLab

A local PyTorch and MuJoCo workbench for an embodied fruit-fly nervous-system model. The fly appears on the left; an activity inspector on the right shows neural population summaries. The web experiment runs the **full imported MaleCNS identified-neuron graph: 166,700 neurons and 25,582,938 directed connections**.

Every imported neuron integrates at 0.1 ms. The CPU reference uses float64; optional Apple GPU execution uses explicitly labeled float32. Raw synapse counts, transmission delays and refractory dynamics follow an explicitly versioned model. Slow computation produces slow motion; the runtime never prunes neurons, skips integration steps, or silently lowers precision. Measured wiring does not supply complete physiology or a complete muscle map. This is an experimental model, not a validated digital organism. See [brain fidelity, continuation and numerical verification](docs/BRAIN_FIDELITY.md).

## Setup and run

Use Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Node.js 22:

```bash
git clone https://github.com/hiyuantang/flylab.git
cd flylab
uv sync --python 3.12
(cd frontend && npm ci)
uv run python scripts/download_data.py
PYTHONPATH=backend uv run python -m flylab.full_connectome
./scripts/dev.sh
```

The three MaleCNS source tables total approximately 1.1 GB. The atomic importer writes `data/full/`; choose another `--output` directory to rebuild without overwriting it. The app requires this full artifact. Missing data or incompatible saved state produces an explicit error; there is no reduced-circuit fallback.

Open [the development app](http://127.0.0.1:5178). FastAPI binds to port 8000 and Vite to 5178, both on loopback. The launcher installs absent dependencies. Stop with Ctrl+C. Stop only this project's previous processes if its ports are occupied.

For a single-server build:

```bash
(cd frontend && npm run build)
PYTHONPATH=backend uv run uvicorn flylab.api:app --host 127.0.0.1 --port 8000
```

Open [the built app](http://127.0.0.1:8000). Multiple tabs share one local controller; this is not an authenticated multi-user service.

## Workbench controls

- **Run / Pause / Step** advance synchronized brain and body time. Step computes 20 ms, including 200 neural updates. The footer reports simulated time, computation time and measured speed. Pacing settings are ceilings, not speed guarantees.
- **Save state / Resume saved** preserve and restore the entire neural state, pending spikes, RNG and physical integration state. Restore opens paused. Normal shutdown also saves, and startup resumes paused. Reset explicitly starts a new episode; forced termination can lose changes since the last save.
- **Environment / Sensory cue** control uniform or spatial odor input. Vision, hearing, wind, touch and joint-position signals are configurable in **What the fly senses**. Sensory tuning remains assumed. Scene, environment and sensory reconfiguration explicitly reset the episode.
- **Brain activity → 3D network** places 139,662 neurons at measured soma locations and colors them by live simulated firing rate. Enlarge the view, click a point or search a MaleCNS ID, and inspect incoming/outgoing partners, synapse counts and directed links. Neurons without coordinates remain searchable. The display uses soma points and graph links, not reconstructed axon shapes.
- **Brain activity → Anatomical groups** explores every imported neuron through superclass, cell type, soma side and assigned column. Search groups or body IDs, inspect live rates/voltages, and compare exact internal/incoming/outgoing connection and synapse counts. Missing annotations remain explicit. **Interventions** retains the six coarse intervention populations and bounded 500 ms pulses or soma clamping. See [group membership and API](docs/ANATOMICAL_GROUPS.md).
- **Data & model → Brain execution** switches the live brain between CPU float64 and experimental Apple GPU (MPS) float32. Switching pauses, backs up the original state in `data/execution-backups/`, and transfers the complete state without resetting time. GPU selection persists in saved state; unavailable hardware produces an error rather than a silent fallback. MuJoCo, sensors, training workers and the isolated paper experiment remain on CPU.
- **Body and muscles** show 69 connected anatomical segments, 70 hinge degrees of freedom, 42 actuated leg joints and 84 effective muscle channels. Supported motor-neuron rates drive finite-force actuators; MuJoCo computes motion and contacts. No gait oscillator or posture tracker controls the web experiment. Successful walking, flight and feeding are not established.
- **Camera / Replay** provide orbit, zoom, muscle and skeleton overlays, segment inspection and recorded display frames. Replay does not rewind the backend. Use Resume saved for actual state continuation.

## Scenes and sensing

Choose Laboratory, Kitchen, Living room, Bedroom or Garden. Scene geometry is shared by rendering, collision physics and head-mounted visual rays. Furniture and terrain have real dimensions; the fly can fall from its starting surface. Overview locates it at room scale, Surface shows its surroundings, and Fly view returns to millimeter scale. Furniture is static and rooms are cutaways. See [scene architecture](docs/SCENES.md).

The sensory panel displays two 16 × 8 grayscale ray-sampled eye views, odor, antennal sound, wind/gravity, foot contact and joint position. Inputs reach 6,091 annotated visual sensory neurons, 114 auditory Johnston's-organ neurons and 475 wind/gravity Johnston's-organ neurons, alongside olfactory and leg sensory populations. Vision pools brightness by eye; retinal columns, color, receptor tuning and recognition remain unresolved. See [sensing assumptions](docs/SENSING.md).

## Data and biological boundaries

The MaleCNS importer retains every annotation with a non-null superclass and every induced connection, without top-k selection or weight thresholds. The local artifact has 124,177,617 synapses. It excludes 44,877 unclassified annotation rows and reports crossing edges separately; segment rows are not interchangeable with identified neurons. Counts remain int64 and source hashes are recorded in `data/full/manifest.json`.

The current Shiu-derived LIF profile uses 20 ms membrane decay, 5 ms synaptic decay, 2.2 ms refractory periods, 1.8 ms transmission delays and 0.275 mV per signed synapse. ACh is positive; GABA, histamine and default glutamate are negative. Unknown/modulatory effects default to zero fast current while those neurons and anatomical edges remain present. Receptor-specific actions and graded transmission are not resolved.

The expanded bridge with pretarsal and peripheral mechanics maps 438 of 815 annotated brain/VNC motor neurons to 154 of 186 muscle channels. The remaining 377 have explicit missing-target or missing-mechanism records. Type names are measured; same-side assignment, fixed-axis force projections, muscle strength and rate-to-force conversion remain assumptions. Older saved states retain their legacy mapping until explicitly upgraded. See the [complete neuromuscular inventory and limitations](docs/NEUROMUSCULAR_MAPPING.md). Body geometry derives from a female NeuroMechFly specimen. See [mechanics](docs/PHYSICS.md), [asset provenance](docs/ASSETS.md), and the mapping in **Data & model**.

## Paper reference experiment

The independent reference uses the paper's **full FlyWire female v630 graph**, not MaleCNS. It runs no-input, sugar and sugar + bitter trials without training and measures MN9 outputs:

```bash
PYTHONPATH=backend uv run python scripts/reproduce_paper.py --download --duration-ms 1000 --seed 42
```

This downloads about 90 MB of pinned source tables. **Data & model → Run full-graph reference** runs the same protocol after downloading. A recorded single-seed trial showed sugar-evoked MN9 activity and suppression with bitter input. It does not reproduce the paper's full trial statistics or all behavioral predictions. [Exact protocol, results and limitations](docs/BRAIN_FIDELITY.md).

The CPU reference passes Brian2 state/spike checks. Custom Metal kernels execute the complete graph on MPS without unsupported sparse tensor operations. GPU float32 passes the small Brian2 scheduling tests, but full-graph trajectories differ from float64 and GPU execution is not necessarily faster. No precision downgrade is automatic. See [GPU implementation and measured limits](docs/GPU_EXECUTION.md).

## Physical learning

**Training** uses cross-entropy search over six neural population gains, sensory gain and motor gain. All anatomical neurons, edges and counts remain intact. Rewards come from MuJoCo balance, motion and effort; completed runs include a zero-muscle ablation and held-out perturbation checks. Applying a compatible full-graph policy restores its saved scene, senses and mechanics and starts a new episode. It does not replace the anatomical wiring.

This is engineering parameter optimization, not yet biological synaptic plasticity or demonstrated learned walking. Older synthetic bandit and posture utilities remain for historical tests; the web controller rejects them. The Gymnasium `FlyLab-Locomotion-v0` environment supports direct `action_mode="muscle"` control with 84 excitations for independent policy research. Its observations include physical state and sensory samples; physics advances 20 ms per action.

## Organization and validation

- `backend/flylab/full_connectome.py`: MaleCNS import and complete graph loading.
- `paper_dynamics.py`, `paper_experiment.py`: verified neuron dynamics and pinned reference experiment.
- `body.py`, `neuromuscular.py`, `senses.py`, `scenes.py`: mechanics, annotation-based coupling and world sensing.
- `simulation.py`, `live_state.py`, `api.py`: synchronized integration, continuation and local service.
- `physical_training.py`, `env.py`: physical reward optimization and Gymnasium interfaces.
- `frontend/src/components/`: body/brain views, sensing, training and reference results.
- `tests/`, `docs/`: deterministic checks, provenance and validation records.

```bash
PYTHONPATH=backend uv run pytest -q
(cd frontend && npm run build)
```

Tests cover dynamics against Brian2, graph direction/count integrity, spike-delay and body-state continuation, physical effects, training isolation and API behavior. Optional dataset checks skip when bulk data are absent. See [validation records](docs/VALIDATION.md).

Further work includes unclassified boundary interpretation, cell-type physiology, retinal routing, complete muscle/tendon mapping, validated locomotion, biological plasticity, and long-run GPU precision validation and performance optimization. MuJoCo is not end-to-end differentiable through the current PyTorch interface.

## License and sources

Original code and documentation: [Apache-2.0](LICENSE), copyright 2026 Yuan Tang and FlyLab contributors. Third-party materials retain their licenses. MaleCNS data is CC BY 4.0; NeuroMechFly assets are Apache-2.0; the Shiu reference code and protocol adaptation retain their MIT notice. Datasets, checkpoints and build outputs are ignored by Git. See [third-party notices](THIRD_PARTY_NOTICES.md).

Primary sources: [MaleCNS downloads](https://male-cns.janelia.org/download/), [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9), [NeuroMechFly](https://github.com/NeLy-EPFL/flygym), and [MuJoCo muscle modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#muscles).

Compound-eye mode uses 857/852 measured optical directions and individual facet input for 3,793 visual neurons. The MaleCNS-to-optical-template registration remains unvalidated; UV, polarization and ocelli are absent. [Sensory implementation, evidence and limits](docs/SENSING.md).
