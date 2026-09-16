# Sensor and contact audit

## Corrections

The `geometry-v3` spatial sensor profile fixes three coordinate errors while
retaining the complete neural graph and existing neuron identities:

- Spatial odor now depends on 3D receptor-to-source distance. Previously, an odor
  source far above or below the fly had the same intensity as one at its height.
  The 8 mm Gaussian field is still an assumed concentration model, not fluid transport.
- Relative wind uses velocity at each antennal sampling point, computed from the
  MuJoCo positional Jacobian and all generalized velocities. The previous root
  translation-only calculation omitted airflow caused by rotation and joint motion.
- The gravity proxy uses the model's gravity vector and distinguishes inversion
  from upright. Zero gravity produces no tilt drive. Conversion into a pooled JO
  drive remains an uncalibrated hypothesis; this does not reconstruct JO mechanics.

Invalid, zero and non-finite sampling intervals now fail before any state changes.
The environment's odor readout reuses the exact sensor frame supplied to the brain.
The sensory form refreshes whenever server settings change, including switches,
source coordinates and contact profiles, while preserving edits when settings are
unchanged. Preparing a laboratory standing trial now resets its virtual sound/cue
source to laboratory coordinates instead of retaining a previous room's source.

## Contacts and neural routing checked

Regression tests verify tarsal-only foot loads, exclusion of body and upper-leg
support from foot support, world-force direction for either contact order,
wall friction versus wall-normal force, and passive support without a root pin.
They also verify mirrored leg-angle derivatives, eye direction geometry, sensory
ablation, and that uncertain JO classes and visual interneurons receive no invented
external sensory input. The imported olfactory class currently contains 2,639
`cb_sensory` neurons; the audit found no interneurons in that direct input group.

Remaining biological gaps are explicit: tactile input is pooled by annotated leg,
not by measured individual hair receptive fields. Non-foot and self-contact
mechanics are simulated, but comprehensive neural touch encoding is absent.
Additional tactile nerve labels, receptor-specific directional tuning, campaniform
load feedback, haltere gyroscopic encoding, odor receptor chemistry, UV and
polarization are not comprehensively reconstructed. Existing evidence does not
justify arbitrary assignments to fill these gaps. Optical self-occlusion and
acoustic propagation through obstacles are also omitted.

## Compatibility and validation

`legacy-v2` retains previous spatial behavior. **Upgrade spatial sensors** pauses,
backs up, and saves a versioned upgrade without resetting or advancing the brain
or body. Old snapshots still restore their original sensor behavior. New standing
trials choose `geometry-v3` and `tarsal-contact-v1` explicitly.

Tests cover 3D translation invariance, height attenuation, receptor velocity
against finite differences, inverted and zero gravity, invalid intervals, backup
integrity, and exact continuation under the new profile. The existing eye,
proprioception, contact, API, scene, training and checkpoint tests are also run.
Browser validation covers the upgrade and form synchronization on the local app.

```sh
PYTHONPATH=backend uv run pytest -q tests/test_spatial_sensors.py tests/test_senses.py tests/test_retina.py tests/test_standing.py tests/test_locomotion.py
```
