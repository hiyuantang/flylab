# Neuromuscular mapping

The `muscle-routing-v5` bridge routes 454 annotated MaleCNS motor neurons into 164 of the body's 196 effective muscle channels. Of those connected neurons, 436 have supported named target identities, 14 have tentative identities, and four are assigned only to a known muscle family. These qualitative categories are not probabilities; every mechanical force path remains approximate. This is a functional approximation, not a reconstructed neuromuscular connectome or evidence of successful walking. All anatomical neurons and recurrent edges remain in the simulation, including motor neurons without body outputs.

## Complete inventory

The installed annotations contain 815 motor neurons: 708 `vnc_motor` and 107 `cb_motor`. Coverage includes both brain and VNC outputs. The v2 profile maps 266 neurons; v3 adds 42 named long-tendon motor neurons; v4 adds 130 neurons projecting to 96 named peripheral channels; v5 adds 16 neurons using ten additional channels. The v4 profile remains available with its original 438-neuron, 186-channel mechanics.

| Target region | Mapped / total |
| --- | ---: |
| Front legs | 123 / 135 |
| Middle legs | 86 / 116 |
| Hind legs | 99 / 130 |
| Abdomen | 0 / 214 |
| Wing/thoracic flight | 64 / 67 |
| Mouthparts | 54 / 67 |
| Neck | 14 / 44 |
| Halteres | 14 / 16 |
| Antennae | 0 / 13 |
| Head `rm` / thoracic `xm` classes | 0 / 13 |

The 361 unresolved outputs comprise 288 non-leg outputs without a supported route (including unresolved muscle identities), 53 leg neurons without resolved muscle identities, and 20 femur-reductor neurons whose muscle action is unknown in the cited atlas. Raw class labels are retained where their interpretation is unresolved.

## Evidence and approximations

Motor IDs, type names, leg subclasses and side annotations come from [MaleCNS v1.0](https://male-cns.janelia.org/download/). Named muscle functions are interpreted using [Azevedo et al. 2024, supplementary methods, Table A1 and Figures A2–A17](https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf). This transfers anatomical roles across specimens; it does not supply this male's measured muscle attachments.

The new routes cover coxa promotion, remotion and adduction, plus sternotrochanter and tergotrochanter extension. Tests check joint geometry and actuator torque signs on all six legs. Tibia flexion/extension and tarsus depression/levation were reversed relative to the rig's coordinates in the legacy profile; v2 corrects both pairs.

Same-side muscle assignment remains an assumption. Conflicting instance-side or informative exit-nerve annotations prevent a new route. Unknown coded neuron names are never matched by ordering or proximity.

The main leg joints use 84 fixed-axis antagonistic Hill-type actuators. Biological insertion geometry, moment arms and muscle-specific force calibration are absent. Remotor/abductor output is split equally between two axes; that split is uncalibrated. Motor rates are pooled per channel, scaled and clipped into excitation. Consequently, neuron coverage is not a biological-fidelity percentage, and these routes do not guarantee locomotion.

## Pretarsal mechanism

The exact labels `ltm MN`, `ltm1-tibia MN` and `ltm2-femur MN` route to six additional Hill-type actuators, one per foot. Each pulls a spatial tendon across a hinged paired-claw mechanism. Tendon length and muscle force determine joint torque; a passive spring returns the hinge when excitation stops. Claws have collision geometry shared with the Three.js renderer. Joint motion is integrated, not prescribed.

This is a **distal transmission approximation**. The [long tendon](https://flybrain.inf.ed.ac.uk/en/blog/releases/ontologies/fbbt/fbbt_00058136_v6/) extends from the femur to the pretarsus; the simulated cable includes only its distal insertion mechanism. Separate proximal muscle bellies and their force summation are pooled into one channel. Claw geometry, a 0–1.1 rad hinge range, 20 µN force parameter, added mass, spring and damping are uncalibrated. Pulvilli adhesion and validated grip/wall walking are absent.

## Named peripheral mechanisms

The v4 model adds force-driven rostrum, haustellum and paired labellum joints; elastic pharyngeal/valve and crop-duct wall elements; a head yaw joint; wing feathering; and haltere stroke/trim joints. Each new muscle has its own excitation and Hill-type force channel, a fixed-tendon transmission, joint limits and passive elasticity. Telemetry exposes activation, force and the mechanical interpretation per channel. No joint trajectory or wingbeat animation is prescribed.

[McKellar et al. 2020](https://elifesciences.org/articles/54978), Table 3, supports primary mouthpart positioning actions. Internal wall displacement follows that paper's insertion hypothesis; the model has no fluid flow, feeding or digestion. [Cui et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11398398/) identifies crop-innervating enteric motor neurons; the added duct element is an effective elastic load, not a reconstructed gut.

Named wing, haltere and tergotrochanter targets use the [MANC motor-muscle matching table](https://cdn.elifesciences.org/articles/96084/elife-96084-supp3-v1.csv) accompanying [Cheong et al.](https://elifesciences.org/articles/96084). Matching uses names, never cross-specimen body IDs. TH1/TH2 neck identities also follow [Nern et al. 2025](https://www.nature.com/articles/s41586-025-08925-z). Satellite TTM and some haltere matches are tentative in the reference.

**Identified target does not imply known mechanics.** Wing/haltere axis projections and the neck yaw reduction are engineering hypotheses, not measured insertions. In particular, assigning shared feathering or spread/elevation actions to steering muscle groups does not reconstruct their distinct sclerite linkages. These routes make their force paths executable for further calibration, not biologically validated. DLM/DVM use a quasi-static Hill approximation: asynchronous stretch activation, thorax deformation, phase-dependent steering and aerodynamic forces remain absent. Wing movement therefore does not establish flight.

All added masses, moment arms, force capacities, springs and ranges are uncalibrated. Detailed mouthpart apodemes and cervical linkages are absent. The remaining 288 non-leg unresolved records must not be interpreted as 304 known muscles awaiting implementation: many have only coded neuron identities. The v5 FNM2 route uses the published cross-midline match with an explicitly approximate transmission. Coverage measures implemented routes, not scientific fidelity.

## Additional routes and confidence in v5

| MaleCNS types | Added neurons | Modeled target | Identity evidence |
| --- | ---: | --- | --- |
| CvN4, CvN5, CvN6, CvN7 | 8 | VL1, pooled per side | Gorko et al. 2024, Figure 4h directly identifies the muscle for these named types. |
| FNM2 | 2 | Contralateral AD neck adductor | Cheong Supplementary file 3, tentative identification with original author score 2/5; the axon crosses the midline. |
| MNwm35 | 2 | iii4 wing muscle | Cheong Figure 6B and Supplementary file 3, putative identification with original author score 1/5. |
| MNhm42, MNhm43 | 4 | hb1 **or** hb2 haltere muscle | The reference identifies the family but explicitly cannot distinguish the two muscles. Separate channels retain this ambiguity. |

[Gorko et al. 2024](https://www.nature.com/articles/s41586-024-07222-5), Figure 4h and Extended Data Figure 7, provide the neck anatomy. The v5 VL1 muscle has an effective pitch-down force path. FNM2 acts on the opposite side to its soma, through a tentative yaw projection. MNwm35 uses the existing third-axillary force approximation; the haltere family channels use an assumed trim projection. These are finite-force Hill actuators, not prescribed movements or target-pose controllers. Their axes, force capacities, moment arms and elastic constants are uncalibrated; the neck proprioceptive convergence reported by Gorko is not reproduced by this routing change.

**Data & model → Inspect motor sources & confidence** shows all 815 motor neurons, searchable by ID, type, region, muscle and source. Connection status and identity confidence can be filtered independently. Each expanded record includes the source link, figure/table locator, identity rationale, mechanical assumptions, target side and any original MANC author scores or matching caveats. Export includes the same fields. A score of 5 in the reference is an ordinal author assessment, not 100% certainty. In particular, MNhm42's high source score does not resolve hb1 versus hb2, and the UI labels it as muscle-family-only.

The curated MANC evidence is in `backend/flylab/assets/motor_target_evidence.json`, with attribution, original source URL and SHA256. It preserves the source's unresolved and tentative entries. The imported MaleCNS annotation file and neural graph are not modified.

The source checks did not resolve the 53 coded/untyped leg targets, the mechanical action of the 20 femur reductors, or the bulk of the abdominal targets. Cheong et al. explicitly report unresolved abdominal motor identities. Antennal/retinal neuron labels and regional membership alone do not establish individual muscle insertions or action signs, so no new outputs are fabricated for them.

## Versioning and continuation

The v5 upgrade pairs `muscle-routing-v5` with `appendage_model=peripheral-v2`, appending ten actuators and one head-pitch joint while preserving old actuator ordering. The v4 upgrade pairs `muscle-routing-v4` with `appendage_model=peripheral-v1`; v3 retains `pretarsal-v1`. Baseline constructors retain v2 and the original body; old checkpoints keep their stored profile (or v1 if absent). Training rejects mismatched body/mapping profiles, and policy loading retains the evaluated mechanics.

In **Data & model**, adding supported and tentative motor routes pauses the simulation and backs up the prior state under `data/mapping-backups/`. All neural tensors, gains and clocks are preserved. Existing physical positions, velocities, muscle states and applied forces transfer by name; added joints start at their spring reference and added actuators start relaxed. This is a physical model migration, not exact continuation of unchanged mechanics. Removing joints from a live body is rejected. Subsequent checkpoints support exact continuation within that model.

## Reproduce and inspect

```sh
PYTHONPATH=backend uv run python scripts/audit_motor_mapping.py
PYTHONPATH=backend uv run pytest -q tests/test_motor_mapping.py tests/test_pretarsus.py tests/test_peripheral_mechanics.py tests/test_live_state.py tests/test_api.py
```

The [machine-readable audit](results/motor-mapping.json) records annotation SHA256, the manifest's graph fingerprint, every mapped transmission, and every unresolved motor ID with its reason. The app provides region totals, searchable missing connections and mapping export. The audit reads annotations without advancing a brain; mechanical tests validate local directions, not full-range anatomical moment arms or animal behavior.

## Previous v3 validation — 2026-09-12

The targeted integration run passed 37 tests with one GPU-dependent skip, covering pretarsal forces, tendon shortening, spring return, named-state migration, exact checkpoint continuation, API behavior, training, sensing and scenes. The existing 13 mapping tests also passed. The production frontend build passed with its existing large-chunk warning.

The live MPS animal upgraded at 0.00 simulated seconds. All nine neural tensors, existing joint positions/velocities and existing muscle activations matched exactly; the six added activations started at zero. A 20 ms full-graph step produced 10,486 spikes and finite mechanics. Its long-tendon neurons did not generate muscle excitation in that step; this is not evidence of spontaneous gripping. The saved paused animal was restored after verification. See [migration evidence](results/pretarsal-migration.json).

## Peripheral v4 validation — 2026-09-12

The targeted suite passed **56 tests with one GPU-dependent skip**. It covers every added channel's neuron-to-excitation and MuJoCo force path, proboscis displacement, elastic wall return, finite scene resets, 186-channel RL action compatibility, exact nonzero-state checkpoint continuation, and API rollback after a failed save. These are implementation checks, not physiological validation. The frontend production build passed; its existing Three.js chunk-size warning remains. Browser checks confirmed the v4 coverage totals, expandable 96-channel force table and articulated body rendering without console errors.

The live MPS migration at 0.00 s preserved all nine neural tensors, RNG state, pending spikes, gains, clocks, and existing physical positions, velocities and muscle activations exactly. New channels started relaxed. A 20 ms full-graph step produced 10,486 spikes in approximately 1.0 s wall time with finite body state; the added peripheral channels received zero excitation during that brief input condition. The saved paused state was restored afterward. [Migration and execution evidence](results/peripheral-migration.json) includes the reference-table SHA256 and graph fingerprint.

## Motor v5 validation — 2026-09-14

The targeted motor, physical-training, continuation, coupling and API checks passed across the focused runs. All four API migration cases pass for v4/v5 and serial/pipelined execution, including rollback after a failed final save. New-channel checks verify neuron-to-excitation routing, actual MuJoCo torque, VL1 pooling, pitch-down direction, contralateral adductor routing, retained ambiguity, and unchanged existing mappings.

The live paused full-graph animal was upgraded to v5 at 0.00 simulated seconds. All nine neural tensors, random-generator state, graph fingerprint, gains, execution settings, environment, clocks, existing joint state and existing muscle state matched the saved v4 state exactly. The old 186 buffered commands and their timestamps were preserved; ten new channels and the head-pitch joint started relaxed. The prior live checkpoint remains in `data/mapping-backups/pre-motor-v5-20260914.pt`. See [state comparison](results/motor-v5-migration.json).

The frontend production build passed with the existing large-chunk warning. Chrome/Playwright checks at 1440×1050 and 390×844 verified the live upgrade, source links, identity/mechanics explanations, family-only ambiguity, confidence filtering, the original 1/5 author score, complete 454-record connected mapping export, and empty-search results. There were no page/console errors, framework overlays or horizontal page overflow. The Browser plugin was unavailable; the installed Chrome browser was used without installing browser dependencies.

An isolated clone of the saved full graph completed one 16.67 ms cycle in 0.863 s wall time, retaining all 166,700 neurons and 25,582,938 directed connections. It produced 2,596 spikes with finite neural/body state and a 196-channel command buffer. New muscle activations remained zero during this one-cycle input condition; it does not demonstrate spontaneous neck movement, flight or walking. The live saved animal was not advanced. See [full-graph smoke check](results/motor-v5-full-graph.json).
