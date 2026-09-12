# FlyLab

A local scientific workbench with a moving fly on the left and a linked neural activity inspector on the right. PyTorch simulates neurons and trains selected synapses; MuJoCo computes muscle activation, force, joint motion and contact; React Three Fiber renders the body state.

This first implementation has two explicitly different models. The embodied experiment uses a **synthetic 96-unit reference circuit** and an articulated NeuroMechFly body with assumed muscle mechanics. The measured-data probe uses **actual MaleCNS v1.0 connectivity** with assumed LIF physiology. The measured graph is not yet wired to the body. Neither mode is a validated reconstruction of a living male fly.

## Run locally

Install Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Node.js 22 LTS with npm, then clone the repository:

```bash
git clone https://github.com/hiyuantang/flylab.git
cd flylab
./scripts/dev.sh
```

Open http://127.0.0.1:5178. The backend uses http://127.0.0.1:8000. The launch script installs Python 3.12 dependencies and frontend packages when they are absent. From an existing checkout, run `./scripts/dev.sh` in its root directory. Both processes bind only to loopback. Stop with Ctrl+C. If either port is in use, stop this project's previous process before starting another; do not kill unrelated applications.

Manual setup:

```bash
uv sync --python 3.12
cd frontend
npm ci
cd ..
PYTHONPATH=backend uv run uvicorn flylab.api:app --host 127.0.0.1 --port 8000
# In another terminal, from the repository:
cd frontend
npm run dev
```

For a single-server build, run `npm run build` in frontend, then start the backend; it serves the built site and meshes at port 8000. Rebuild and restart after frontend changes.

## Experiments

- **Run / Pause / Step / Reset** advance or reset simulation time. Reset clears neural state, muscles, body pose, interventions and the recording, while preserving learned parameters.
- **Environment** selects uniform cues for conditioning or a spatial Gaussian odor field sampled at the moving antennae. Changing environments pauses and resets the body. **Sensory cue** selects odor A, B or no odor. Spatial steering uses an explicit approach/avoid decoder; navigation success is not established.
- **Brain activity** shows mean absolute modeled unit activity. Select a region in the visualization, list or selector. Stimulation injects a bounded 500 ms pulse; silencing clamps that population. A paused pulse advances when simulation time advances.
- **Muscle activation** groups 84 effective Hill actuators by leg and torque direction: blue positive, orange negative. Hover for activation and total force in µN. Foot indicators show ground contact. The overlay represents modeled activation, not reconstructed muscle volumes.
- **Body movement** uses 69 connected anatomical segments, 70 hinge degrees of freedom and 42 actuated leg joints. Recorded walking references guide finite-force muscles through an explicit controller; MuJoCo computes body motion and ground contact. Self-collision prevents nonadjacent legs and wings from passing through the body. Wing surfaces use smaller convex sections and constrained passive hinges; reset resolves initial intersections. Walking passes short stability checks. Flight, grooming and biological muscle calibration remain unimplemented.
- **Replay** becomes available while paused. It displays the latest 400 streamed body/neural snapshots (up to about 20 seconds at 1×); it does not rewind backend state. The signal plot retains the latest 300 integration steps (6 simulated seconds). Export includes both buffers and model provenance.
- **Camera** supports orbit/zoom, perspective/top/side, muscle overlay, a connected-skeleton overlay, follow mode and reset. Select a body segment to inspect its attachment.

## Reinforcement learning

The website’s training task is a two-cue contextual bandit. At each trial the reference brain observes a synthetic odor and chooses approach or avoid. Reward is +1 for approaching the rewarded odor or avoiding the other; otherwise −1. The task does not reward physical locomotion.

REINFORCE updates the 256 MB-to-descending synaptic parameters. All other weights and the connection pattern stay fixed. Each update uses 32 independent trials. Training operates on a copy of the current brain, leaving the live experiment unchanged until **Apply trained brain** is clicked. A frozen two-cue evaluation reports exact action probabilities; it is not an out-of-distribution navigation benchmark.

Completed and stopped runs save tensor checkpoints under data/checkpoints. Applying or loading a checkpoint resets the body and transient neural state; learning remains. Restore untrained reference brain resets learned parameters explicitly. The current task uses an engineering optimizer, not a validated dopamine-dependent biological plasticity mechanism.

### Physical locomotion environment

The separate Gymnasium environment rewards physical progress toward a target, penalizes muscle effort and falls, and ends on success or a time limit. Use it from a source checkout after installing dependencies:

```python
import gymnasium as gym
import numpy as np
import flylab.env  # registers FlyLab-Locomotion-v0

env = gym.make("FlyLab-Locomotion-v0", action_mode="synergy", max_steps=500)
observation, info = env.reset(seed=42)
for _ in range(500):
    action = np.full(6, 0.4, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

`synergy` accepts six leg drives in [0, 1]. `muscle` accepts 84 independent excitations and bypasses the reference controller. Both return 338 normalized observations describing joints, muscle activation, body motion, foot forces, target bearing, odor and controller state. Actions advance 20 ms of physics. This headless interface is ready for PyTorch policies; the example is a constant-drive baseline, not a trained policy. Physical locomotion training is not yet connected to the website’s training panel.

See [body mechanics and assumptions](docs/PHYSICS.md) for the control hierarchy and validation limits.

## Measured MaleCNS data

The download script obtains three public Janelia flat-connectome tables, approximately 1.1 GB total:

```bash
uv run python scripts/download_data.py
PYTHONPATH=backend uv run python -m flylab.connectome --seed 10001 --limit 256
```

The imported subgraph includes DNp01 body 10001 and up to 255 directly connected, annotated neighbors ranked by summed incident synapse count. Eligibility requires a non-null superclass. All measured edges induced by those IDs are retained, with original integer synapse counts. The importer records excluded crossing edges; it does not assume those inputs are biologically absent.

The local import contains 256 neurons, 13,839 connection rows and 98,830 synapses. It excludes 616,223 boundary connection rows. The raw connection table has 151,856,684 segment-to-segment rows; this is not a neuron count. Source URLs, sizes, SHA256 hashes, selection rule and neurotransmitter predictions are in data/manifest.json.

The probe uses sparse PyTorch recurrence with 1 ms LIF integration. Its normalized efficacy, thresholds, decay, reset, refractory behavior and transmitter sign mapping are assumptions. It treats only predicted GABA as inhibitory; glutamate and other transmitters need receptor-specific modeling before physiological interpretation. With default settings a DNp01 pulse can spike only the stimulated cell. That is an outcome of the assumed dynamics, not proof that downstream neurons do not respond in vivo.

Data and checkpoints are gitignored. The UI works with the reference model when bulk data are absent. The importer streams Arrow batches and makes two passes without loading the complete graph as a dense matrix.

## Architecture

| Module | Responsibility |
|---|---|
| backend/flylab/neural.py | Synthetic recurrent rate circuit and trainable synapses |
| backend/flylab/body.py | Connected anatomical rig, muscle actuators and walking controller |
| backend/flylab/arena.py | Synthetic odor sampling and target bearing |
| backend/flylab/env.py | Gymnasium locomotion and direct muscle actions |
| backend/flylab/assets/ | Pinned anatomical parameters and numeric gait references |
| backend/flylab/simulation.py | Time integration, interventions, sensory/body bridge |
| backend/flylab/training.py | REINFORCE, evaluation and checkpoints |
| backend/flylab/connectome.py | Versioned measured data import and sparse LIF probe |
| backend/flylab/api.py | Local HTTP controls, WebSocket snapshots and static hosting |
| frontend/src/components/Scene.tsx | 3D rendering of backend body poses |
| frontend/src/components/Brain.tsx | Selectable schematic brain and stimulation controls |
| frontend/src/components/Telemetry.tsx | Recorded signals, replay and muscle inspection |
| frontend/src/components/Training.tsx | Training and saved-brain workflow |
| frontend/src/components/DataPanel.tsx | Provenance and measured-wiring experiment |

The server is a single local workbench. Multiple tabs share simulation and training state. It is not an authenticated multi-user deployment. Heavy training runs independently of UI rendering. The 1×/2×/4× setting is a requested stepping multiplier, not a real-time performance guarantee. End-to-end differentiability through MuJoCo is not provided.

## Validation

```bash
PYTHONPATH=backend uv run pytest -q
cd frontend
npm run build
```

Tests cover stimulus expiry, motor silencing and muscle decay, physical joint motion, determinism, reset semantics, learning and reversal, checkpoint round trips, preservation of fixed connections, sparse edge direction, data count integrity, API validation and live streaming. Browser QA covers desktop and mobile layout, live controls, training/apply/evaluation, measured-data probing and replay. See docs/VALIDATION.md.

## Scientific development still required

1. Map MaleCNS motor and sensory IDs to experimentally supported muscle and receptor targets; preserve species, sex and specimen provenance.
2. Replace the synthetic reference circuit with selected validated sensorimotor circuits, then expand. Build explicit boundary input models for subsets.
3. Calibrate cell-type and receptor-specific physiology, graded versus spiking transmission, delays, and compartmental dynamics where supported.
4. Replace effective actuators with calibrated muscle geometry, tendon routing and force laws. Add active neck, proboscis, wings and abdomen mechanics.
5. Add vision and biological sensory encoders beyond the synthetic odor field and contact feedback; train embodied policies with held-out environments and perturbation benchmarks.
6. Implement and compare local dopamine-modulated plasticity against engineering RL, without treating task success alone as biological validation.

## Sources and attribution

- MaleCNS project and CC-BY data: https://male-cns.janelia.org/ and https://male-cns.janelia.org/download/
- NeuroMechFly: https://neuromechfly.org/ and https://github.com/NeLy-EPFL/flygym
- Body geometry, joint frames and masses derive from a female NeuroMechFly specimen. Visual and physical segments share transforms and a uniform scale; muscle routing and control remain assumptions. See [asset provenance](docs/ASSETS.md) and the bundled Apache 2.0 license.
- MuJoCo muscle model: https://mujoco.readthedocs.io/en/stable/modeling.html#muscles
- Fly connectome modeling precedent: https://www.nature.com/articles/s41586-024-07763-9
- Dopamine and memory dynamics: https://www.nature.com/articles/s41586-024-07819-w

## License

FlyLab's original code and documentation are licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Yuan Tang and FlyLab contributors.

Third-party materials retain their own licenses. MaleCNS data is CC BY 4.0; the bundled NeuroMechFly meshes are Apache-2.0; DM Sans and IBM Plex Mono fonts are SIL OFL-1.1. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, source versions, modifications, and dependency notices. Bulk datasets, trained checkpoints, installed dependencies, and build outputs are excluded from Git.
