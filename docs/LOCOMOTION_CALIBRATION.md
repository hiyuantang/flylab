# Locomotion calibration and research protocol

The subsequent bounded parameter search and rejected fits are documented in
[Bounded calibration results](CALIBRATION_FIT.md).

## Current conclusion

The measured-graph model has working motor transmissions and some restoring
reflex responses, but coordinated walking and active balance are not validated.
The standing baseline is supported mainly by assumed passive joint elasticity.
An optimizer can improve its standing score simply by suppressing muscle output;
that candidate must not be presented as learned balance.

The first experiment used all **166,700 neurons and 25,582,938 directed edges**,
MPS FP16, 960 neural steps/s and 60 muscle commands/s. Every four-second trial
computed 3,840 neural steps, including the motor-disconnected control. No graph
pruning, gait tracking, root pinning, reward training or live-state changes were
used. These results are exploratory tests of one model configuration, not
independent biological replicates or a standalone performance benchmark.

## Reproduction

From the repository root:

```sh
PYTHONPATH=backend .venv/bin/python scripts/calibrate_locomotion.py \
  --output data/calibration/new-run --stage all --seconds 4
PYTHONPATH=backend .venv/bin/python scripts/probe_reflexes.py \
  --checkpoint data/calibration/new-run/input.pt \
  --output data/calibration/new-run/reflexes-14mv.json --pulse-mv 14
PYTHONPATH=backend .venv/bin/pytest -q tests/test_calibration.py
```

The runner copies the input checkpoint, resets an isolated experiment, archives
source code, and records hashes, engine versions, physiology, body parameters,
clock settings, initial-state fingerprints and per-frame measurements. Output
directories must be new. Bulk traces and checkpoints stay under ignored `data/`.
Each completed condition is saved immediately; interrupted runs retain results.

The first run is in `data/calibration/2026-09-13-baseline/`.
Its archived source records the executed revision; the maintained runner also
reports held-out activation and differences from passive support. Compact,
versionable evidence is in [results/locomotion-calibration.json](results/locomotion-calibration.json).

## Motor and sensory interface

A fixed-pose diagnostic clamped each mapped motor neuron's output rate to 100 Hz,
one at a time, then evaluated the actual MuJoCo generalized actuator force.
All **438 mapped neurons** produced forces in their assigned directions without
unexpected directly actuated degrees of freedom. **377 motor neurons remain
unmapped**. This verifies internal transmission consistency, not biological
muscle insertions, force calibration or free-body joint displacement.

Six knee-coordinate perturbations verified mirrored flexion/extension signs.
The current four-type FeCO encoder routes **154 proprioceptors**; **876 annotated
proprioceptors have unresolved tuning/routing in this encoder**. Its right-front
`SNpp50` flexed-angle group is empty in the imported annotations. No mirrored
replacement neuron was invented. These gaps do not imply the real animal lacks
the corresponding sensation.

## Matched standing experiments

All conditions began with identical body-pose and neural-voltage fingerprints.
A 3 µN lateral thorax push lasted from 2.0 to 2.1 simulated seconds. The body
used `stance-elastic-v1`; passive spring stiffness and motor gain are assumptions.
Support requires feet to carry the body, excluding thorax and upper-leg support.

| Condition | Foot-supported time after 0.5 s | Peak tilt after 0.5 s | Net displacement |
|---|---:|---:|---:|
| Full feedback | 96.2% | 7.70° | 0.375 mm |
| Proprioception disabled | 99.0% | 8.88° | 0.217 mm |
| Touch disabled | 98.6% | 4.39° | 0.066 mm |
| Both feedback channels disabled | 99.5% | 3.76° | 0.053 mm |
| Muscle output disconnected | 99.5% | 3.76° | 0.053 mm |

An exploratory sensory log-gain sweep over 0, −1 and −2 selected −2 by post-push
support, then peak tilt. That setting produced zero muscle activation. With an
opposite-direction held-out push, its physical result matched the disconnected
control. **It was not applied to the live fly.** The short push is also an easy
test for the passive body; harder perturbations and longer trials are needed.

## Reflex and stepping assays

With the body fixed, a 100 ms warmup established an identical sensory background.
Each available leg/receptor group then received a 50 ms additive 14 mV current
pulse, followed by 100 ms observation. The assay measured target firing and
incremental knee torque against a matched no-pulse trace. It estimates torque
at steady muscle activation; it does not measure free-body balance.

All 23 available groups increased their target firing. Nine produced net knee
torque consistent with the assumed restoring response, one (`RH extending`)
produced the opposite sign, and thirteen had no detectable incremental local
knee torque in this observation window. The remaining group was unmapped.
The earlier 7 mV threshold-level assay is retained separately; silent output at
threshold alone is insufficient evidence of an ineffective neural pathway.
These results depend on background, current amplitude, observation window and
the assumed rate-to-force model. They do not justify automatically flipping a
receptor's biological tuning.

Four unpushed trials compared no descending command with constant `DNg100`
amplitudes 0.4, 1.2 and 2.4, beginning at 0.5 s. None produced a candidate swing
meeting the diagnostic requirements: unloading, ≥0.03 mm clearance, ≥0.1 mm
displaced landing, and at least three command intervals. Loaded-foot movement
was recorded separately. **Displacement did not establish walking.** Contact
thresholds and sample resolution can miss small or fast steps; these are
screening criteria, not a validated biological gait classifier.

## Research contribution and next gates

A possible paper question is: **Which physiological and peripheral assumptions
allow a preserved connectome to produce effective closed-loop motor control?**
This is a research hypothesis, not an established novelty or publication claim.

1. Test the weak and opposing reflex responses across input rates, backgrounds
   and delays; inspect the corresponding annotated premotor pathways. Resolve
   receptor and muscle assignments using independent anatomical evidence.
2. Fit uncertain parameters to independent motor/reflex measurements. Separate
   calibration data from held-out behavior; retain neuron IDs, edges and
   provenance. Compare active feedback with matched motor-disconnected and
   sensory-ablation controls across perturbations and starting poses.
3. Require repeatable foot lifting, load transfer, propulsion, and comparisons
   with recorded fly kinematics before accepting a walking controller. Test
   sensitivity to timestep, precision, muscle strength and passive elasticity.
4. Once locomotion is established, evaluate navigation and dopamine-dependent
   plasticity with learning-disabled controls. Behavioral training and fitting
   unknown simulator parameters must be reported separately.

The current runs support an engineering diagnostic result and expose calibration
failures. They do not support claims of reconstructed innate walking, a faithful
digital animal, or biological learning.

## Scientific context

- [MaleCNS](https://male-cns.janelia.org/) supplies anatomical connectivity and
  annotations; it is not a complete specification of embodied physiology.
- [Vaxenburg et al., Nature (2025)](https://www.nature.com/articles/s41586-025-09029-4)
  trained artificial controllers to imitate measured fly locomotion. This is
  an important behavioral/engineering comparison, not evidence that imported
  MaleCNS wiring alone walks.
- [MANC functional annotation, Figure 59](https://elifesciences.org/reviewed-preprints/97766v1)
  predicts opposing FeCO effects on knee motor circuits. This motivates the
  restoring-response assay; individual MaleCNS sensory tuning remains inferred.
