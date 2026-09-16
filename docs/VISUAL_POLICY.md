# Sensor–action transformer

FlyLab supports a separate, fully trainable visual engineering controller alongside the frozen MaleCNS connectome and its adapter. Neither controller replaces the other's weights or anatomy. This is the first sensor/action pair for a robot-training platform; language is absent.

## Model and interfaces

The default model contains 1,386,708 parameters: width 128, four pre-normalized transformer encoder layers, four attention heads, and two decoder layers with five action queries. Dropout is disabled. All weights and gradients use explicitly reported float32, on MPS when available, CUDA otherwise, then CPU. This precision choice belongs to this small model and does not change MaleCNS execution.

Each sensor has a named, serialized `SensorSpec`: sample count, channel count, token budget, and geometry. The model accepts a dictionary of tensors `[batch, history, samples, channels]`. Missing or unexpected sensors fail validation. Eye tokenizers and a generic vector tokenizer are implemented; adding another sensor requires connecting its sampler to this registry and training a matching schema. No hearing, touch, proprioception, gesture ID, target activation, or robot joint state enters the current policy.

New transformers use **1,024 visual units per eye, grouped into 64 tokens of exactly 16 nearby units**. Their viewing directions are interpolated from the imported right-eye microCT map and its hexagonal facet coordinates ([Zhao et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12488493/)). A dense triangular lattice is interpolated only inside complete adjacent-facet triangles, without extrapolating beyond the measured topology. Deterministic farthest-point sampling in facet coordinates selects 1,024 units. Sampling in facet coordinates rather than uniformly in visual angle retains the measured map's curved footprint and nonuniform angular density approximately.

The left eye mirrors this common right-eye template across the fly's midline. Each eye casts its own rays. The equal counts, idealized bilateral symmetry, interpolation, and grouping are engineering choices, not measured anatomy. Recursive spatial partitioning in the hexagonal sheet assigns exactly 16 units per group; groups are not a rectangular grid in visual angle. Per-unit history and direction pass through a shared learned projection before group averaging. Every unit contributes once; pooling compresses detail. Seven-ray acceptance-cone integration and its assumed 2° ring remain unchanged.

New models serialize `compound-retina-balanced-v3` with `embodied-left/right` sensor geometry. They retain the measured-map directions and add estimated eye-surface origins and anatomical self-occlusion. See [Compound vision](COMPOUND_VISION.md) for research, rendering assumptions, and validation. The previous `compound-retina-balanced-v2` / `anatomical-left/right` schema retains its original world-only ray casting. The earlier rectangular angular layout remains available only for existing `balanced-left/right` schemas, preserving their input ordering and checkpoint behavior.

Existing measured-eye transformer checkpoints retain their original 857 left / 852 right samples and angular pooling. Their serialized schema selects the correct optics during inference and continued training. They are not silently converted; train a new transformer to use the new layout. The measured MaleCNS retina and imported anatomical data remain unchanged.

Eight causal visual frames (at 20 ms intervals) provide temporal context. At episode start, the earliest observed frame is repeated; no future frames enter the input. The model predicts five 20 ms excitation commands for the 84 named leg actuators. Inference replans every 20 ms and executes the first command. Non-leg actuator excitation is zero. Existing MuJoCo muscle activation and force/body integration perform the movement; there is no pose assignment or gesture lookup in inference.

`policy_model.py` owns the schema/tokenization/network; `policy_simulation.py` owns the optical adapter and physical controller; `policy_training.py` owns training and checkpoints. The simulation subclass reuses non-neural physical snapshot scaffolding, but overrides the step completely: it never executes the posture controller. UI snapshots explicitly identify the transformer and show parameters rather than anatomical neurons.

## Training and evaluation

The training recipe is `action-chunk-bc-v1`: direct action-chunk behavior cloning. Physically checked gesture references supply actual excitation commands. The transformer predicts those commands with masked L1 regression over all 84 leg actuators and valid future steps. Padded steps do not contribute. Teacher muscle states are not part of this loss. This follows the action-reconstruction approach used in [ACT's official implementation](https://github.com/tonyzhaozh/act/blob/main/policy.py), without adding its CVAE or changing FlyLab's architecture.

Whole demonstration episodes stay in one split. The default dataset contains 16 randomized placements per enabled gesture for training and four separate placements per gesture for action validation. Seeds are `seed + 2000 + episode` and `seed + 4000 + episode`; left/right cues use paired placements. Training weights follow the configured gesture proportions; validation weights gestures equally. All teacher trajectory windows are retained. Dataset caches record settings, schema, seed, target/training provenance, commands, masks, and episode IDs.

An **epoch** visits all training windows in shuffled order. **Batch size** means windows per optimizer update, not complete physical demonstrations. Adam updates every transformer weight after each minibatch, with gradient clipping at 1. Defaults are 60 epochs, 32 windows per minibatch, and LR 0.0003. For two gestures, 16 demonstrations each and 25 steps, there are 800 training windows, 25 updates per epoch, and 1,500 updates in 60 epochs. There is no scheduler or mixed precision.

Action MAE is evaluated every epoch on separate teacher trajectories. Every ten epochs, at early stopping, and at completion, the learned policy controls fresh physical trials from reset. Evaluation uses four unseen placements per enabled gesture (`seed + 8000 + episode`). A trial passes only when the existing correct-leg, stance-support, body-height, and upright criteria hold for all five final 20 ms frames. The preview shows an actual evaluation episode. The UI reports per-gesture success separately from action error.

The saved inference policy maximizes physical validation success, with lower action MAE breaking ties. Once a physically evaluated candidate exists, an unevaluated epoch cannot replace it. Latest training state is always saved separately. These repeated checks are a **validation** set used for model selection; an additional untouched placement split is needed for final test claims. This separation follows [robomimic's rollout evaluation practice](https://robomimic.github.io/docs/tutorials/viewing_results.html).

This is a standard, measurable baseline, not a claim of a universally best training algorithm. The visual-only controller remains partially observed. If failures appear on recovery states or changed environments, collect appropriate demonstrations and evaluate distribution shift before adding methods such as DAgger. The current reference generator is not an expert for arbitrary recovery states.

## Product flow and persistence

Select **Vision–Action Transformer** in Training. Configure epochs, minibatch size, demonstrations per gesture, duration, learning rate, gesture proportions, and optional early stopping. The default new-model learning rate is 0.0003. Stop requests show immediate feedback and pause at an epoch boundary; Resume retains optimizer, RNG, and current weights in the running backend. Start new discards that in-memory continuation. After a backend restart, selecting a version and continuing restores a complete saved training snapshot. The default is the latest completed update; the custom Continue from menu also supports branching from the best evaluated policy snapshot. Both options create a child version.

Every completed epoch atomically saves both latest and best snapshots to the run's version. Each snapshot contains matching full weights, Adam moment buffers and step counters, the local NumPy sampling-generator state, the cumulative update count, validation loss, and early-stopping progress. The best weights remain the inference entry point. Sensor/action schema, body/sensor settings, target version, parent lineage, training-source and target-route hashes, runtime versions, and results are retained. Unlike MaleCNS adapters, these are full-model checkpoints. Each weight set has approximately 5.5 MB of parameters; matching Adam buffers and separate best/latest snapshots add storage. Each new training run creates a new lineage node, including when initialized from a previous transformer. Connectome adapters cannot initialize transformers, or vice versa.

The Weights page browses both families. Branch deletion checks the exact descendant set and protects the active model. Experiment **Load model** dispatches by checkpoint family. Loading saved transformer weights starts a paused lab scene and persists the selected model for backend restart; that restart begins a new physical trial. It does not claim continuation of the full physical state. Refreshing the page retains the running backend state and persisted UI selections.

### Continuation rules

- Selecting a parent populates seed, duration, and gesture mix; a parent using the same recipe also supplies LR, minibatch size, and demonstration count. These remain editable. For the same recipe, continuation restores Adam first, then applies the requested LR, preserving moment history and step count. Moving from activation-MSE training to action-chunk L1 retains the selected weights but explicitly starts fresh Adam and a new validation monitor. There is currently no LR scheduler.
- With the same seed, restore the local training sample stream. An explicitly changed seed starts a new stream and a new validation definition. Dropout is zero and the current trainer has no stochastic torch operations after isolated, seeded model initialization; process-global torch RNG is not overwritten. Adding dropout, augmentation, or another random sampler requires adding its owned RNG state to this checkpoint protocol.
- Validation and early-stopping history carry over only when their definition matches (settings, enabled cues, seed, horizon, target/training implementation). A changed definition starts a fresh monitor. LR changes alone do not reset Adam or the monitor. Patience counts epochs since the last meaningful improvement, including accumulated small improvements. A run already at its patience limit performs no extra updates unless patience is increased or early stopping is disabled.
- Best and latest have separate complete snapshots; never attach a later optimizer state to earlier best weights. When continuing latest, retain the ancestor's better evaluated snapshot until a new best is found. Training-step counts begin at the selected snapshot; chart epoch numbers are local to the child run.
- Legacy weight-only versions remain loadable. They explicitly report fresh Adam and a new sample stream; the lost history cannot be reconstructed from those files. Future children save the complete format. Existing checkpoint files are not rewritten.
- Inference compatibility is unchanged. Training-source provenance is separate, so the continuation fix does not invalidate existing inference weights. Bitwise reproducibility across different hardware or library versions is not promised.

### Continuation verification

The real training loop was tested with an isolated small transformer schema, physical teacher trajectories, and temporary checkpoints: four uninterrupted updates matched two updates followed by save/load and two more. Saved model tensors, Adam tensors/counters, sampling state, and monitor values matched exactly on both CPU and Apple MPS in these runs. Additional tests cover LR overrides without moment resets, separate best/latest origins, legacy checkpoint fallback, changed seeds, early-stopping carryover, and API dispatch isolation. Seven focused CPU/API checks and the separate MPS equivalence check passed. Related policy, activity, API, gesture, and preview checks passed after fixing the controller-specific request routing and updating a stale preview test double to the current snapshot contract; two existing environment-dependent checks skipped.

The frontend production build passed (with the existing large-bundle warning). Browser-plugin tooling was unavailable, so isolated headless Chrome through bundled Playwright checked `http://127.0.0.1:5178/` at 1600×1100 and 900×1000. Checks covered meaningful page content, no framework overlay, no browser warnings/errors, the legacy notice, latest default, best selection, inherited LR, and the actual outgoing LR/continuation request. A temporary modern-version response fixture exposed the new controls; training POSTs were intercepted. Screenshots were kept outside the repository. This UI test did not start a real training job.

After the local backend restart, health and MPS/float32 execution were verified, the completed training plot was preserved, the new OpenAPI continuation field was present, and SHA-256 checks confirmed that all four existing policy checkpoint files were unchanged. The live experiment restarts from its selected saved model; its physical trial is not resumed.

## Training review: saved v3, 2026-09-15

The current model is a vision-action transformer: no language input or language encoder is implemented. It is not yet a vision-language-action (VLA) model in the [RT-2 sense](https://robotics-transformer2.github.io/). This is an explicit scope choice, not a checkpoint bug.

### Measured diagnostic

Read-only evaluation used `policy-1789419860934720000.pt` on Apple MPS in float32, its recorded body/sensor settings, 25 steps per gesture, and the original fixed validation placements (NumPy seeds 100042 and 100043). No training or checkpoint changes occurred. For the blank-input ablation all eye arrays were zeroed; for the swap ablation the 25 left/right teacher histories were exchanged while retaining each sample's original activation state and targets.

| Validation input | Muscle activation MSE | Change from normal |
| --- | ---: | ---: |
| Original visual histories | 0.000421073623 | — |
| Both eyes zeroed | 0.000421172823 | +0.02356% |
| Left/right histories exchanged | 0.000421089840 | +0.00385% |

Two separate learned-policy rollouts, one per gesture, ran from reset for 25 real physics command steps (0.5 simulated seconds). Neither reached the requested pointing pose under `evaluate_reference`. The requested front foot reached about 0.028 mm for left pointing and 0.029 mm for right pointing, against the existing 0.3 mm success threshold. These are two diagnostic trials, not a statistically estimated success rate. [Raw measurements](results/visual-policy-v3-audit.json).

This result indicates very weak visual dependence on the tested windows and poor behavior on these two trials. It does not isolate whether the limiting factor is optimization, input representation, teacher coverage, partial observability, or the objective. Further training with complete Adam history alone is not established to solve it.

### Resolution

The current recipe directly supervises expert commands, performs a real optimizer update per minibatch, separates entire demonstration episodes, and selects checkpoints using physical gesture success. The original activation-loss loop could reduce average error while barely distinguishing left from right; a run labeled 20 iterations performed only 20 optimizer updates. Preserving Adam fixes continuation but does not itself fix this training objective or insufficient optimization.

The historical results above describe the previous recipe. New action MAE values cannot be numerically compared with its activation MSE values. Existing weights remain loadable, and older checkpoints are preserved. The model architecture and real inference physics remain unchanged.

## References

This is an engineering adaptation, not a reproduction or pretrained derivative of a published model.

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/): transformer action chunks. This implementation omits ACT's CVAE and proprioceptive inputs.
- [π₀](https://arxiv.org/abs/2410.24164) and [π₀.₅](https://www.pi.website/blog/pi05): visual conditioning and continuous action sequences. This implementation omits the language backbone and flow matching; direct regression suits the initial deterministic gesture references.
- [Optical provenance](../backend/flylab/assets/retina/README.md).

## Historical pipeline verification

`tests/test_policy_model.py` checks complete facet coverage, equal eye token budgets, extensible sensor schemas, batch equivalence, finite gradients, physical inference, loss decrease on real sampled optics, exact teacher-command muscle integration, checkpoint round trips, lineage deletion, and pause/resume.

The isolated five-iteration MPS experiment used three 500 ms samples per batch, mixed left/right pointing, random placements, and 0.0003 Adam. Fixed teacher-window validation MSE fell from 0.0075430 to 0.0059107 (21.6%) in 79.7 seconds. Saved-model inference succeeded. These are pipeline checks, not established gesture success. See [recorded results](results/policy-gpu-validation.json).

The integrated Chrome run completed three updates, including Stop/Resume, and reduced fixed validation loss by 15.49%. Its `Visual transformer starter` version loaded in Experiment, accepted a hand cue, advanced muscle/body physics, and remained selected after refresh. Browser console inspection showed no warnings or errors. The 50 Hz command schedule was preserved, but the full scene/sensor/policy/physics loop ran in slow motion (about 0.18× real time in this check); 50 Hz wall-clock execution is not established. Backend regression checks passed (57 tests with two environment-dependent skips), followed by nine focused policy tests and a production frontend build. Existing MaleCNS forward compatibility digest remained unchanged.

## Engineered activity view

Loading a transformer switches the Experiment activity panel to its own 3D
layout. The measured connectome retains its anatomical view. The default policy
view contains 980 points in nine sections: 64 tokens per eye, 128 feature
channels per encoder/decoder layer, and 84 next-command muscle excitations.
These are representation summaries, not a biological neuron count or a drawing
of all 1,386,708 parameters.

Eye-token values are RMS across feature channels. Layer-channel values are RMS
across tokens (or action queries), with final layer normalization included in
the last encoder and decoder sections. Muscle values are sigmoid outputs for
the first action query, before the body's activation dynamics. Hidden colors
are normalized separately per section; muscle colors use the fixed 0–1 range.
Selection displays the unnormalized value and reduction. Lines depict module
signal flow, including encoder memory to the decoders, not attention weights.

Read-only inference hooks in `policy_activity.py` record detached summaries and
cache their CPU serialization per forward pass. They do not capture training
graphs or alter checkpoint schemas, weights, or forward-implementation hashes.
Reset shows an unmeasured state until the next forward pass. Search accepts
point IDs such as `encoder_1:1` or exact actuator names. Points select inline;
blank space clears selection; only the enlarge button opens a dialog.

Validation: 31 policy, telemetry, and API tests passed, with two data-dependent
skips. Tests include measured output values, gradient isolation, and API step/reset
behavior. Browser checks cover search, selection, background deselection, and
the dedicated enlarged view. A 30-forward Apple GPU check on the saved starter policy
measured 1.82 ms without hooks and 2.12 ms with hooks; outputs were identical
in that check. These timings cover the network only, not sensing or physics.

## Behavior-cloning validation, 2026-09-15

The full, unchanged 1,386,708-parameter model trained from random weights on Apple MPS in float32: two pointing gestures, 16 demonstrations per gesture, 25 steps per demonstration, minibatches of 32 windows, Adam LR 0.0003, seed 42, and 60 epochs. Early stopping was disabled for this measured run. The run performed **1,500 optimizer updates** in 296 seconds including demonstration preparation and periodic physical checks. This timing is specific to this machine/run.

The selected checkpoint is `policy-1789445040153777000.pt`, lineage **Pointing · behavior cloning**, epoch 50 / update 1,250. Its eight model-selection rollouts passed, and its held-out action MAE was 0.00093646. The final epoch remains available as the separate latest training snapshot.

An additional test used untouched placement seeds 16000–16007, one second per episode, and the same physical success criterion:

| Gesture | Successful test episodes |
| --- | ---: |
| Human left hand points → left front leg | 8 / 8 |
| Human right hand points → right front leg | 8 / 8 |

Zeroing both eyes raised teacher-trajectory action MAE to 0.00871966 (9.3×); exchanging left/right visual histories raised it to 0.01658548 (17.7×). This supports visual dependence on these inputs. The test establishes performance for the two trained cues within the current randomized laboratory distribution; it does not establish all-gesture performance, unseen environments, recovery behavior, or robustness across training seeds. [Complete measurements and provenance](results/visual-policy-bc-validation.json).

Reproduce physical evaluation with fresh seeds:

```sh
PYTHONPATH=backend uv run python scripts/evaluate_visual_policy.py \
  policy-1789445040153777000.pt --start-seed 16000 \
  --episodes-per-gesture 8 --steps 50 --output /tmp/policy-evaluation.json
```

Backend verification covered direct command targets, future masks, episode-level splits, cached datasets, sustained physical success, changed-objective Adam reset, and exact split/resume equivalence on CPU and MPS. The related API, policy, activity, gesture, and preview suite passed 61 checks with two environment-dependent skips. The frontend production build passed with the existing large-bundle warning. Isolated Chrome verified new epoch/minibatch/demonstration controls, legacy-objective notices, best/latest selection, and LR override requests; those control tests intercepted training POSTs. Actual training and physical testing ran independently on MPS. Existing checkpoint files were preserved.
