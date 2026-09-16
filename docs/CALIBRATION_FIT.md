# Bounded calibration results

## Outcome

No replacement setting passed the tested reflex and support checks. The live
fly retains its previous parameters. This is an exploratory fit of uncertain
simulator parameters, not a calibration against recorded biological waveforms.

All trials preserved 166,700 neurons, 25,582,938 edges, MPS FP16 and the existing
960 Hz neural / 60 Hz muscle clocks. Twelve four-second body trials each computed
3,840 neural steps, with matching initial pose and voltage fingerprints.

## Neural efficacy

The search varied the VNC output log gain, keeping measured connections and
transmitter signs unchanged. Left-leg reflexes selected the setting by
`restoring responses - 2 × opposite responses`; ties favored the smallest
change. Right-leg responses were evaluated separately. These are model-derived
qualitative hypotheses, not independent measured torque targets.

| VNC log gain | Left restoring / opposite | Right restoring / opposite |
|---:|---:|---:|
| −0.50 | 1 / 0 | 3 / 0 |
| 0.00 | 5 / 0 | 4 / 1 |
| 0.15 | 5 / 1 | 5 / 3 |
| 0.30 | 5 / 1 | 5 / 2 |
| 0.50 | 7 / 2 | 7 / 1 |
| 1.00 | 5 / 6 | 7 / 2 |

The original zero log gain won the selection. Amplifying the VNC increased some
responses but introduced wrong-direction responses; suppressing it increased
silence. There were 12 available left-leg groups and 11 right-leg groups. The
unmapped right-front flexed-angle group was excluded from scores, not fabricated.

## Muscle-force scaling

The body search tested force multipliers 0.50, 0.75, 1.00 and 1.25 with unchanged
neural gains, mass, passive springs and friction. At a +3 µN push, 0.75 improved
foot-supported time from 96.2% to 99.0%. However, during the opposite push its
post-push supported fraction fell to 87.0%, versus 96.5% for the original. No
continuous 250 ms supported recovery window was detected before the trial ended.

The 0.50 alternative reduced tilt but also failed opposite-direction support:

| Push | Original post-push support | 0.50 force support | Original / 0.50 peak tilt |
|---:|---:|---:|---:|
| −3 µN | 96.5% | 93.9% | 7.70° / 5.01° |
| +6 µN | 99.1% | 97.4% | 7.80° / 5.01° |
| −6 µN | 98.3% | 93.0% | 10.93° / 5.01° |

The opposite-direction results informed further exploration; they are not an
untouched final test set. Both stronger push directions then checked the 0.50
candidate. Its active muscles and reduced movement did not compensate for lost
support. No calibrated walking controller was obtained. One isolated candidate
swing appeared in a 1.25-strength pushed trial; it did not establish locomotion.

## Acceptance and reproduction

The engineering screen requires nonzero muscle activation, supported-time loss
no greater than 1 percentage point overall or 2 points after a push, peak tilt
increase no greater than 0.5°, loaded-foot travel increase no greater than 5%,
no new body support above 0.01 µN, and retained recovery when the baseline
recovers. These tolerances are screening assumptions, not statistical confidence
or physiological constants. More initial poses and longer runs remain necessary.

```sh
PYTHONPATH=backend .venv/bin/python scripts/fit_calibration.py \
  --output data/calibration/new-fit
PYTHONPATH=backend .venv/bin/pytest -q tests/test_calibration.py
```

The maintained fitter archives its input and source before running, writes each
result immediately, selects on the left side, validates the body candidate, and
reports whether it passes. It never applies a fit automatically. Raw exploration
is in `data/calibration/2026-09-13-fit/`; its source archive was assembled after
the runs. [Compact evidence](results/calibration-fit.json) contains artifact
hashes, parameters, metrics and rejected checks.

## Remaining calibration targets

The current linear sensory-current encoder has a threshold dead zone: isolated
LIF receptors receiving less than the rest-to-threshold current do not fire.
Joint-velocity tuning and cell-type-specific physiology need direct evaluation;
uniform gain changes did not resolve the tested reflexes. A uniform muscle-force
multiplier also cannot represent distinct motor units. Experiments report large
differences in resting firing, excitability and force per spike among fly tibia
motor neurons; the current imported muscle labels do not establish which IDs
correspond to those recorded units. See [Azevedo et al. (2020)](https://elifesciences.org/articles/56754).

The next fit should use independently supported motor-unit identities and
sensory input/output measurements. Reducing motion or increasing the number of
active neurons alone is insufficient evidence of improved biological control.
