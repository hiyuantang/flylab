# Sensory environment

The workbench uses a deterministic, headless sampler in `backend/flylab/senses.py`. MuJoCo world geometry supplies the same furniture, plants and surfaces shown in the scene. Measurements follow the physical head/antenna transforms; they do not depend on the observer's camera. Brain input and training use the same sampler.

## Compound-eye optics

`vision_model=compound-retina-v1` replaces the 16 × 8 grid with **857 left and 852 right measured viewing directions** from the microCT reference accompanying [Zhao et al. 2025](https://doi.org/10.1038/s41586-025-09276-5). The two eyes are independently measured. Their irregular angular spacing is retained, not replaced with a decorative hexagonal mosaic. The source is another specimen, not the MaleCNS animal.

Seven ray samples per direction approximate optical acceptance: the central ray has weight 0.4; six rays on a 2° ring each have weight 0.1. This quadrature and head-to-eye alignment are assumptions. World materials are converted from sRGB to linear visible intensity before integration. R1–R6 uses an assumed broad-visible weighting; R8p and R8y use blue/green material proxies. These are not calibrated rhodopsin response curves. Ordinary RGB does not specify UV reflectance: R7/UV, polarization, ocelli and uncertain spectral types have no invented input. Shadows, fly self-occlusion and photoreceptor adaptation remain absent.

The UI is an angular receptor map: front/side/rear along the horizontal axis and dorsal/ventral vertically. Dot intensity shows an individual facet response; it is not a claim about the fly's subjective experience. Canvas drawing avoids thousands of React elements per update. Optical geometry and neural assignment are cached; every imported neuron and connection remains active in the simulation.

## Retinal routing and spatial accuracy

MaleCNS visual receptor rows lack `assignedOlHex1/2`. Many downstream visual neurons do have those fields. `retina.py` groups each photoreceptor's outgoing synapse counts by the annotated column of same-eye postsynaptic neurons. It accepts a dominant column with at least five synapses and at least 50% of column-assigned evidence; ties are unresolved. These thresholds select external input assignments, not retained brain connections.

**The optical registration is not yet spatially validated.** MaleCNS hex coordinates are centre-aligned to the published right-eye lens grid using shared axis coordinates. The left correspondence uses the nearest independently measured left direction to the mirrored right template. These are explicit cross-specimen registration hypotheses. Equator, axis orientation, peripheral correspondence and the body-frame alignment require anatomical validation. A connectivity-supported column is not itself a validated viewing direction.

The installed-data audit routes **3,793 of 6,091 side-annotated visual neurons** to individual facet signals, without eye-wide averaging. Of 2,298 unresolved inputs, 196 lack column evidence, 468 are ambiguous, 456 fall outside the transferred template, and 1,178 lack a supported spectral channel after spatial checks. These neurons still receive recurrent network inputs and update normally. No CNN replaces the measured visual circuitry. The existing LIF model also treats graded photoreceptors as spiking units; functional vision remains unvalidated.

## Other modalities

| Input | Measurement | Recipient annotations |
| --- | --- | --- |
| Sound | Directional RMS at each antenna, distance attenuation and assumed 250 Hz preference | `cb_sensory`, `JO-*`, `auditory`: 114 neurons |
| Wind/gravity | Relative air velocity projected onto antennal axes and head tilt | `JO-*`, `wind_gravity`: 475 neurons |
| Odor | Uniform concentration or spatial Gaussian field | Annotated olfactory sensory neurons |
| Touch | Normalized leg contact load | Supported tactile annotations |
| Proprioception | Assumed tibia-angle tuning | Supported proprioceptive annotations |

Sound uses analytic RMS across 20 ms to avoid carrier aliasing. Wind input does not supply aerodynamic force. Odor A/B remain indistinguishable in the measured controller. Taste, temperature, humidity and calibrated directional hearing remain absent. The laboratory retains its virtual dark sphere/grid; the sphere is a calibration stimulus without collision geometry.

## Versioning and continuation

The sensor engine retains `scene-senses-v2`; the optical behavior is separately versioned by the saved `vision_model` setting. Missing settings and library defaults retain `legacy-grid-v2` for checkpoint compatibility. Its 16 × 8 images and mean-brightness routing are unchanged. The workbench's **Use compound eyes · preserve neural state** action pauses, saves a backup under `data/sensory-backups/`, changes only the sensor model, and saves the upgraded state. Neural tensors, pending spikes, RNG, clocks and physical integration state are preserved. A failed save rolls back the sensor replacement. This changes future input, not past neural state.

Compound Gymnasium observations append 1,709 broad-visible facet samples plus four antennal values. The baseline body therefore has 2,051 observation values; bodies with additional joints/actuators have larger observations. Legacy observations remain 598 for the baseline body. Request compound sampling with `SensorySettings(vision_model='compound-retina-v1', vision_enabled=True)`. Policies must be evaluated with their stored sensor model.

## Evidence and checks

- [Optical asset provenance, extraction and license](../backend/flylab/assets/retina/README.md).
- `PYTHONPATH=backend uv run python scripts/audit_retina.py` reproduces the [routing inventory](results/retinal-routing.json), including every receptor's reason, evidence fraction and selected facet.
- Tests in `tests/test_retina.py` verify direction counts, acceptance integration, head rotation, deterministic local responses, distinct spatial currents, ambiguous/cross-eye rejection and checkpoint continuation. API tests cover preserving the current animal during upgrade.
- A local 20-sample kitchen benchmark measured approximately 12 ms median per complete sensory sample. This is a CPU sampling benchmark, not full-brain throughput.

## Live validation — 2026-09-12

The upgrade preserved the animal at 2.50 simulated seconds: all nine neural tensors, delayed spikes, RNG, clocks, the complete MuJoCo integration state, graph fingerprint and body fingerprint matched exactly. A subsequent 20 ms MPS step completed with compound input and finite muscle values. The saved paused state was restored after the check. See [migration evidence](results/retina-migration.json).

The targeted retina/senses/API/checkpoint suite passed 23 tests with one GPU-dependent skip; the retina/API/training/scene suite passed 30 with one skip. An additional save-failure rollback check passed. The production frontend build passed with its existing Three.js chunk-size warning. Browser checks verified both angular maps and channel switching without console errors. These checks demonstrate implementation behavior, not biological visual performance or correct anatomical registration.
