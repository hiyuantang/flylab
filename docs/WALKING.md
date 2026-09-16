# Measured-circuit locomotion experiments

The workbench now supports targeted descending-neuron stimulation and opposing leg feedback. **Coordinated walking is not validated.** Movement, ground-contact changes, and staying upright are separate measurements; tipping or sliding is not a successful gait.

## Live control loop

`Walking circuit` applies constant input to the annotated DNg100 or DNb08 neurons for three simulated seconds. DNg100 has two mapped cells; DNb08 has four. The input passes through all 166,700 neurons and 25,582,938 directed connections, the existing motor-neuron decoder, Hill actuators, and MuJoCo. There is no imposed oscillation, target pose, direct motor-neuron stimulus, force assist, or gait reference in this path. Stop removes external drive; existing neural and physical dynamics continue. Active commands survive live checkpoint save/restore (format v4); Reset clears them.

The new `feco-opponent-v1` sensory profile samples actual femur–tibia opening and signed angular velocity. Tests check the velocity against geometric finite differences on all six legs, including the mirrored right side. Root-side and entry-nerve annotations determine the leg.

| Type | External signal | Evidence status |
| --- | --- | --- |
| SNpp50 | Flexed position | Stabilizing tuning inferred from extensor-biased output |
| SNpp51 | Extended position | Stabilizing tuning inferred from flexor-biased output |
| SNpp39 | Extension velocity | Stabilizing direction inferred from flexor-biased output |
| SNpp41 | Flexion velocity | Stabilizing direction inferred from extensor-biased output |

The opposing motor effects are described in [Marin et al., Figure 59](https://elifesciences.org/reviewed-preprints/97766v1). These assignments are hypotheses, not recordings of individual MaleCNS cells. Position curves and the 20 rad/s velocity scale are uncalibrated. Club vibration receptors and unresolved proprioceptors receive no invented angle input; they remain fully simulated. The legacy tactile model still pools foot force per leg. Old checkpoints retain their previous sensory encoding. Enabling FeCO feedback backs up the full live state without resetting neurons or the body.

## Physiology experiment

[Pugliese et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/) identify recurrent VNC circuits that generate motor rhythms under tonic descending input. Their preprint uses a rectified-tanh rate model with cell-size-dependent excitability. This does not demonstrate full embodied walking.

`cpg_rate.py` independently implements that equation for an **offline full-CNS experiment**, using the complete existing CSR graph. It explicitly uses FP32, Heun integration, and measured segmentation volumes. Normalization uses the median volume of all annotated VNC neurons; missing volumes use that median and are counted. This differs from the study's graph selections, normalization population, and adaptive integration. It does not replace the live FP16 LIF brain or its saved state.

## Reproducible evaluation

```sh
PYTHONPATH=backend .venv/bin/python scripts/probe_walking.py \
  --seconds 3 --drive 1.2 --motor-log-gain -2 \
  --output /tmp/walking-lif.json
.venv/bin/python scripts/download_neuron_volumes.py
PYTHONPATH=backend .venv/bin/python scripts/probe_walking.py \
  --rate --drive 250 --motor-log-gain -2 --output /tmp/walking-rate.json
```

Trials run on isolated copies. Reports retain graph identity, precision, feedback assumptions, motor excitation, circuit rates, displacement, upright fraction, and foot-contact transitions. Compare `--drive 0`, `--feedback off`, and the two descending targets. The three-second LIF trial at 1.2× drive and motor gain exp(-2) stayed within 30° of upright, but its forward-motion integral was negative: this is not evidence of forward walking.

Before claiming walking, establish repeatable, supported forward steps across longer trials; verify swing trajectories and inter-leg coordination, distinguish slip from stance, and show appropriate loss of behavior under neural interventions. Muscle gains and neuron physiology remain major calibration gaps. The existing dopamine rule changes selected mushroom-body connections; it is not a validated locomotor learning rule.

## Current validation results

All trials below retained the entire graph and used motor gain exp(-2), with three simulated seconds per trial. These are exploratory single-seed trials, not a validated model comparison. Runtime varied under concurrent local work and is not a benchmark.

| Trial | Upright within 30° | Forward integral (mm) | Outcome |
| --- | ---: | ---: | --- |
| [LIF, 1.2× input, FeCO feedback](results/walking-lif-low-force.json) | 100.0% | -0.496 | Walking not established |
| [Rate, input 250, FeCO feedback](results/walking-rate-low-force.json) | 33.3% | -7.849 | Walking not established |
| [Rate, input 500, no feedback](results/walking-rate-open-loop.json) | 100.0% | -0.128 | Walking not established |
| [Rate, input 250, reference volume scale](results/walking-rate-reference-volume.json) | 100.0% | -0.065 | Walking not established |
| [Rate, input 1000, reference volume scale](results/walking-rate-strong-drive.json) | 9.4% | -0.852 | Walking not established |

The reference-volume trials use 998,603,428.5 voxels, the median of the study’s 4,310-neuron MaleCNS annotation table, without selecting or dropping any of our neurons. The downloaded full-CNS size artifact contains 166,678 measured volumes and 22 missing values, matched by neuron ID. Its source generation, ordered-ID hash, and output hash are retained beside the ignored data file.

Stronger rate-model input recruited motor output but caused loss of support. Lower-input trials largely remained quiescent downstream of the descending neurons. These results do not reproduce the paper’s rhythm findings or establish a locomotor learning mechanism.
