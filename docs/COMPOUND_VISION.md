# Compound vision: biological basis and simulation limits

## What a fruit fly can see

A compound eye samples many small, overlapping angular regions. It does not supply hundreds of complete camera images. The measured microCT eye maps show nonuniform angular sampling: resolution varies across the visual field. Near the eye equator, the reported coverage reaches less than 10° across the front midline and about 155° toward the rear, leaving less than 20° binocular overlap and an approximately 50° posterior blind region. These are approximate study measurements, not constant rectangular bounds at every elevation. Flies can also redirect their retinas by about 15°; this simulation does not implement that movement. [Zhao et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12488493/)

The NeuroMechFly v2 camera approximation uses about 270° total horizontal coverage. That is a model calibration, not grounds to crop the newer measured eye map to the same number. [NeuroMechFly v2, 2024, Extended Data Fig. 3](https://www.nature.com/articles/s41592-024-02497-y)

Seeing toward the rear side is different from seeing directly behind the body. A leg, antenna, or body segment can intercept a line of sight when its position puts it in the visual field. This is a geometric consequence; the simulation's specific self-visibility is not a behavioral validation of what flies perceive or recognize about their own limbs.

## The 1,024-unit adaptation

The transformer uses 1,024 visual samples **per eye** and 64 groups of 16 samples. This count is a computational design choice. Directions are interpolated in the measured right eye's hexagonal facet coordinates; the left template is mirrored. This retains approximately the measured directional density and coverage without inventing a full 360° field. Exact bilateral symmetry is an engineering assumption. The imported measured eye arrays remain unchanged.

The previous surface model uses `compound-retina-balanced-v3`, serialized as `embodied-left/right` sensor geometry. It keeps the v2 directions and grouping, and changes their optical rendering:

- Each facet starts at an estimated point on its eye mesh. Its central direction locates the surface from the mesh center; a 0.00001 mm outward offset avoids a numerical self-hit.
- Seven acceptance-cone rays leave each facet. Their origin and orientation move with the head and eye mesh.
- Rays intersect scene geometry and visible anatomical meshes. The observing eye itself is excluded; hidden collision proxies are excluded. Other body parts remain visible and occlude farther objects.
- Hand stimuli and the optional calibration sphere use the same origins and nearest-surface depth, so they cannot paint over an intervening leg.
- Existing model schemas retain their previous optical behavior. Training a new model is required to adopt the changed input distribution.

Surface placement registers optical directions from one specimen to a different anatomical mesh. That registration is an approximation. Materials currently supply RGB intensity proxies; photometric shading, optical refraction, wing transmission, calibrated spectral sensitivity, UV, polarization, ocelli, active retinal movement, and full photoreceptor dynamics are not simulated. The grayscale angular plot is a visualization of receptor inputs, not a reconstruction of a fly's subjective experience.

## Nine-pixel patches

The first nine-pixel schema uses `compound-retina-balanced-v4`, with `patch-left/right` schemas: 1,024 units per eye, nine ordered grayscale pixels per unit, eight history frames, and 64 tokens per eye. Each pixel is one ray. The nine values are retained independently through the observation window and learned tokenizer; the mean is used only for legacy summary displays. The retinal canvas renders the individual samples. This is a bio-inspired engineering camera, not a claim that real ommatidia produce 3 × 3 images.

Each local patch uses row-major offsets in an orthonormal tangent plane. Its sample spacing is one third of the nearest central-axis angular separation. Corner rays stay within half that separation, avoiding overlapping neighboring patch footprints. The v2/v3 measured-map central directions and 64 groups of 16 are unchanged, preserving their denser frontal/equatorial angular sampling and natural rear blind region.

Lens origins are registered independently onto the outward half of each anatomical eye mesh. Angular coordinates map to ±85° in surface azimuth and elevation, leaving a five-degree boundary margin. Ray/mesh intersection locates each origin; ray directions remain measured-map directions. This registration spreads origins toward the exposed front, rear, dorsal and ventral edges instead of using viewing axes as surface coordinates. The exposed half and margin are modeling assumptions, not a segmented corneal measurement or biologically validated lens positions. They must not be confused with expanding visual coverage to 360°.

Existing checkpoint schemas keep their original rays, sensor values and tokenizer. Loading an old version or continuing its training does not silently adopt patches. New input distributions require training a new transformer.

Validation: `tests/test_patch_vision.py` checks independent pixel signals and gradients, bounded patch footprints, preserved central directions, outward placement/margins and legacy schemas. `tests/test_vision_geometry.py` checks the static 3D display origins against the sensor implementation for all five optical versions. A direct comparison with the pre-change source produced exactly identical rays, full sensor payloads and predictions for all four legacy layouts.

A local CPU comparison with the palm stimulus (12 warm samples, stationary body) measured median sensor times of 63.0 ms for v3 and 69.5 ms for v4, about 10% more in this scene. Ray count rises from 14,336 to 18,432 per two-eye frame (29%). Raw policy observations grow ninefold. These timings are not an end-to-end training benchmark or evidence of improved task performance.

## Validation of the previous surface model

`tests/test_embodied_vision.py` checks lateral rear coverage and a near-horizon rear blind region, surface origins following body rotation, direct interception by legs and antennae, farther hits when test leg meshes are hidden, and foreground depth passed to a distant visual stimulus. `tests/test_balanced_retina.py` checks counts, grouping, measured-map density, and old schema behavior.

In the default standing pose, a local ray audit detects front/middle leg segments and antennal aristae from the eye surfaces. The v3 left template spans approximately −4.6° to +151.5° among samples within 10° of the horizon; the right template mirrors it. These are sampled-model values, not new biological measurements. A two-eye sample took about 0.05 seconds in a local CPU check without hands; runtime depends on scene and hardware.

## Exposed-surface placement (current transformer default)

`compound-retina-balanced-v5` (`surface-left/right`) keeps all v4 viewing axes, nine ray directions and 64-token grouping. It changes sampling origins. The previous ±85° mapping used an arbitrary half-eye cutoff and left a large part of the exposed mesh unsampled.

The new placement samples the eye mesh by triangle area and rejects candidates occluded by the head mesh along the lateral exterior direction. Four offset probes erode the head boundary by 0.004 mm. Farthest-point selection covers the remaining surface; a bounded density weight derived from measured viewing directions favors denser angular regions while protecting sparse-region coverage. A spatial partition match assigns the selected origins to existing axis indices without changing the optical axes. This is a deterministic registration heuristic, not measured ommatidial positions or an exact anatomical cornea segmentation.

In the reference pose, 870 of 1,000 mesh vertices per eye pass the exposure test. All 1,024 selected origins pass it. Every independently checked exposed mesh vertex is within 0.016 mm of a sampling origin. `tests/test_exposed_eye.py` checks coverage, exposure and unchanged optical directions. The buried attachment faces are omitted from the curved retinal display; they are not represented as a large dark retinal region. Previous checkpoint schemas retain their original origins.

## Curved retinal display

The Vision panel shows two independently rotatable eye surfaces with live grayscale samples. It displays only the eyes, with no whole-body anatomy, direction cloud or flattened angular chart. Rotate, zoom and reset controls are available for each eye in both inline and enlarged views. Surface shading supplies a shape cue; sample brightness comes from the selected live receptor channel without lighting modulation.

`scripts/build_vision_geometry.py` exports the actual eye mesh, central sampling origins and display positions in head coordinates. For nine-pixel models, nine display markers are spread around each shared facet origin and projected back onto the eye mesh, in the same row-major order as the nine live values. This is a display convention: the training rays still originate at one shared position per unit. Historical center-origin layouts use a display-only projection onto the eye mesh. No trained inputs or weights change when rotating the view.

`tests/test_vision_geometry.py` checks all optical versions against sensor geometry, nine-value ordering, mesh indices and surface attachment. Eye-to-mesh registration remains an engineering approximation rather than validated anatomical lens locations.
