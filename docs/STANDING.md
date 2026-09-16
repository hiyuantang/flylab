# Standing baseline

The **Prepare standing trial** button saves a backup of the current life in
`data/standing-backups/` and starts a new, paused laboratory experiment. Press
**Run** to advance it. The full 166,700-neuron, 25,582,938-edge MaleCNS graph stays
on the existing device and precision. No reinforcement learning is performed.

## What supports the body

`stance-elastic-v1` adds assumed passive elasticity of 20 µN·mm/rad to the 42
active leg hinges. Spring rest angles are the fixed initial reference pose;
there is no running target tracker, gait oscillator, root pin or upright assist.
Passive tarsal springs remain unchanged. The motor firing-rate conversion uses
`exp(-2) ≈ 0.1353` times the original excitation scale. These values are mechanical
calibration assumptions, **not measured MaleCNS physiology**.

Identified FeCO receptors receive actual joint position/movement; corrected foot
contacts feed the existing annotated sensory routes. Motor neurons alone supply
active muscle commands. The quiet trial disables visual, auditory, wind and odor
cues, while every neuron and connection continues to run. It cancels previous
interventions and resets neural/body state and learning through the normal reset.

Elasticity supports the fly even without neural activity. This is a physical
standing baseline, not evidence of reconstructed neural balance. Biological
resistance reflexes motivate testing sensory feedback: [Akitake et al. (2015)](https://www.nature.com/articles/ncomms8288)
measured FeCO-dependent tibial muscle responses in Drosophila. That study does
not validate this model's spring constants, motor gains or receptor tuning.

## Correct contact accounting

Earlier `foot_feedback` counted any leg segment touching the world. A collapsed
fly could therefore appear supported on its feet. The corrected signal includes
only tarsal and pretarsal segments. Support diagnostics transform full contact
forces into world coordinates, accounting for contact ordering and friction;
wall-normal force is not interpreted as vertical support. Thorax, abdomen and
proximal-leg loads are separate. Legacy sensory checkpoints retain their old
signal through `legacy-leg-v1`; the standing trial uses `tarsal-contact-v1`.

The instantaneous support indicator requires at least three loaded feet, 80–120%
of body weight through feet, under 5% through other segments, and tilt below 30°.
It measures neither sustained balance nor neural causation.

## Validation on 2026-09-13

Both isolated trials received a 3 µN lateral push for 0.1 s at simulated second 2.
Metrics use every 60 Hz sample after 0.5 s of settling. Complete reports are in
`docs/results/standing-neural.json` and `standing-passive.json`.

| Trial | Simulated duration | Support criterion met | Minimum loaded feet | Maximum tilt | Other support | XY drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full brain, FP16 MPS | 30 s | 97.5% | 5 | 8.1° | 0 µN | 1.55 mm |
| Passive body, zero muscle excitation | 10 s | 98.9% | 5 | 4.6° | 0 µN | 0.13 mm |

The neural body still twitches and drifts. These trials support elastic-assisted
stance; they do not establish active stabilization, general terrain balance,
walking or biological fidelity. The durations differ, so drift values are not a
matched-duration estimate of neural benefit. Reported wall times include local
contention and should not be used as isolated performance benchmarks.

Reproduce without modifying the saved life:

```sh
PYTHONPATH=backend uv run python scripts/probe_standing.py --seconds 30 --push --output /tmp/standing-neural.json
PYTHONPATH=backend uv run python scripts/probe_standing.py --mode passive --seconds 10 --push --output /tmp/standing-passive.json
PYTHONPATH=backend uv run pytest -q tests/test_standing.py
```

Chrome inspection confirmed that the old pose moved and collapsed onto upper
legs; the new live trial held its body clear of the floor. Tests cover passive
support without a pin, collapsed-body detection, force orientation, sensory
compatibility, full-graph preservation, backups and exact checkpoint continuation.
