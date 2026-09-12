# Validation record

Current automated result: **54 tests passed** in one full run; TypeScript and production build passed. The Three.js bundle still produces Vite's advisory size warning.

## Full-brain physiology and continuation — 2026-09-12

The web workbench defaults to the full imported MaleCNS graph with `shiu-2024-linear-delay-v1`, float64 state and 0.1 ms neural steps. Synthetic/posture controller commands are rejected. A missing graph disables execution rather than supplying a smaller fallback.

New tests compare PyTorch voltage, synaptic current and exact spike sequences against Brian2 under recurrent excitation/inhibition, delayed transmission and refractory input. Other tests prove invariant event delivery and batching, exact full-state/body checkpoint continuation, checkpoint rejection when neurons are missing, synchronized clocks, and API save/restore. Small deterministic fixtures are verification instruments; the deployed simulation retains the full graph.

The full 127,400-neuron FlyWire paper reference ran without training. A 1-second, seed-42 sugar trial produced MN9 rates of 73/58 Hz; adding bitter input reduced them to 6/11 Hz. The no-input control produced zero spikes. Running the protocol from the website reproduced the same spike counts and rates. This is one stochastic trial per condition, not a replication of all published results. [Protocol and recorded result](BRAIN_FIDELITY.md).

The embodied 166,700-neuron, 25,582,938-edge model completed a first 20 ms step in 0.477 wall seconds with 10,486 spikes and one excited muscle channel. Save → advance → restore returned to 0.02 simulated seconds and the original spike count. Graceful server restart automatically recovered that same full state, paused. The saved file is approximately 34 MB and stays outside Git.

The final site was inspected at 1280×720 and 390×844. Fixed-step controls, visible simulated/compute clocks, save/restore feedback, full-graph provenance and paper experiment execution worked. Mobile document width equaled its 390-pixel viewport. A stale dynamic-import error occurred in an already-open tab during a production rebuild; reloading the matching final assets resolved it, with no new runtime errors observed. Viewport overrides were reset after verification.

PyTorch 2.14 detects MPS on the M1 Pro but rejects float64 and sparse CSR. CUDA is absent. The validated CPU backend remains active; no reduced-precision GPU substitute is enabled.

The records below describe earlier component and historical controller benchmarks. Their synthetic gait/posture results do not establish current full-brain locomotion.

Verified locally on 2026-09-12 with Python 3.12, PyTorch 2.14, MuJoCo 3.13, React 19.2 and Three.js 0.180. Dependency versions are locked in uv.lock and frontend/package-lock.json.

At the initial repository setup, the source was also checked against an export of the staged source with no bulk dataset or existing frontend dependencies: backend tests reported **9 passed, 1 skipped** (the optional measured-data integrity test), and a fresh `npm ci` followed by the production build passed. Python source and wheel distributions build successfully and include the Apache-2.0 license metadata, LICENSE, and NOTICE. Frontend builds retain the font/icon license files and emit bundled dependency license texts at `assets/THIRD_PARTY_LICENSES.md`.

## Articulated body checks

The expanded body was checked at `http://127.0.0.1:8000/` in the Codex in-app browser at desktop 1536×1024, mobile 390×844 and the default panel size. Run/pause, perspective/top views, camera reset, skeleton display, eye selection (showing its head attachment), environment switching and changing live muscle/contact values worked. Mobile content remained within the viewport. The production build and attributed assets loaded successfully. Earlier server/frontend version mismatches produced errors during development; the final matching build did not reproduce them.

New automated checks cover all 69 parent attachments during motion, 70 hinge degrees of freedom, 84 muscle actuators, two-second upright forward walking, pulling muscle forces, ground loads, decay after silencing, moving antennal odor sampling, environment reset semantics and Gymnasium action/observation/seed/time-limit contracts. The current observation contains 598 values, including 260 new sensory channels. Collision tests check a separated reset pose, wing contact response and sampled walking penetration below 0.015 mm. All 39 mesh hashes were independently matched to the pinned upstream Git tree. Before the wing-joint update, a wheel built from the source distribution was extracted into an isolated temporary directory; it included all 39 meshes and their license, instantiated the body, and returned a valid observation after a physical step.

Successful navigation and general locomotion robustness remain unvalidated. Direct muscle control is exposed through Gymnasium; the web training panel provides physical policy search alongside the odor decision task. See [PHYSICS.md](PHYSICS.md) for assumptions and limitations.

The collision update was also inspected in desktop and mobile views, including Run/Pause and perspective/top/side camera controls. Wings remain above the abdomen in the inspected side view. The favicon uses the title's bug icon and is served successfully by both local servers. No fresh browser errors appeared during the final checks.

## Full graph and physical learning

The full annotated-neuron import preserved 166,700 neurons, 25,582,938 directed connections and 124,177,617 synapses as integer count CSR. Fixture tests verify duplicate aggregation, pre→post direction, boundary accounting, silence without input, persistent spiking state, reset and unmapped motor targets. The optional local full-artifact test verifies count totals and exclusion accounting. Body tests additionally cover independent parameter configurations and recorded-gait joint envelopes. API tests cover controller switching and mechanics validation.

Neural gain sensitivity probes (2, 8 and 20; 100 ms generic olfactory stimulus) produced no spontaneous spikes. They produced neural propagation but no motor firing during that particular protocol. No experimental recordings were fitted; this is sensitivity characterization.

The `stand-pose-v2` run (seed 43, two generations, four candidates, 25 steps = 0.5 s) improved posture-controller reward from 0.31843 to 0.32293. It passed the nominal task and two of three short stress checks; the +10% mass case failed the height criterion. The zero-muscle comparison failed the pose/height criteria. The same-budget MaleCNS run changed its neural gains and produced nonzero muscle output, but failed stance in the nominal and all three validation conditions. That MaleCNS evaluation predates the `lif-step-rate-v2` correction and is historical, not a measurement of the current neural dynamics. These results do not demonstrate learned walking or calibrated physiology. Older checkpoints without a reward version used weaker upright-only criteria and cannot establish maintained pose.

Reproducibility regressions verify identical firing rates across differently partitioned integration calls, preserved learned gains after physiology edits, and graph reloads when the requested artifact directory changes. Five-step workbench trajectories exactly match independent training rollouts for both physical controllers, with legacy uniform and explicit spatial environments. Malformed checkpoint applications leave live state untouched. Cancellation during checkpoint serialization publishes no policy and removes temporary files. Physical telemetry now consistently records mushroom-body activity in the MB trace.

The updated checkpoint flow was checked on localhost:8000 in the in-app browser at 1280×720 and 390×844. A legacy neural evaluation displayed its model-version notice; Apply stayed disabled until metadata loaded. Applying the policy selected the 166,700-neuron controller and Step advanced to 0.02 s without console errors or a framework overlay. The mobile panel stayed within the viewport. The experiment was left paused at reset. Long-horizon neural learning was not rerun for this bug-fix validation.

The complete UI was checked in the in-app browser on localhost:8000 at 1536×1024 and 390×844. Controller selection loads the measured graph, Step advances its neural/body state, and direct motor-population stimulation produced 5,459 total spikes with nonzero muscle activation in the inspected 20 ms step. The recorded-gait joint profile applied successfully and all joint ranges were inspectable. Physical training completed through the UI; a saved evaluation was reviewed and its checkpoint applied to the posture experiment. No fresh browser errors or horizontal overflow appeared. Brain shapes remain schematic population displays, not anatomical neuron geometry.

## Functional evidence

- Backend tests cover physical joint motion, nonzero muscle force, deterministic reference trajectories, stimulus expiry, motor-population silencing and muscle decay, and preserving learning while resetting transient state.
- Training tests verify improved expected reward for both odor-A and odor-B reward mappings; only the selected plastic parameters change. Fixed connectivity and sensory projection weights are bitwise unchanged. Checkpoints round-trip into a separate brain instance.
- A 100-update / 3,200-trial reference run, seed 42, improved exact two-cue expected reward from −0.0179 to 0.99993. Approach A changed from 0.45998 to 0.999969, and approach B from 0.47791 to 0.00003697. This establishes learning in the synthetic contextual bandit, not biological fidelity or physical navigation learning.
- Sparse-probe tests establish pre→post orientation, no spontaneous spikes with zero input, deterministic stimulation, original integer count preservation and explicit excluded boundary edges.
- The real data importer streamed the complete 151,856,684-row segment-level connection table. The 256-neuron probe preserves 13,839 rows / 98,830 synapses and reports 616,223 excluded crossing rows. Default DNp01 stimulation produced six spikes in the stimulated neuron and no downstream spikes; no propagation is fabricated.
- API tests cover schema rejection, step controls, live WebSocket state and serving the attributed research meshes.
- TypeScript and Vite production build pass. Three.js creates a large lazy-loaded rendering chunk; Vite reports an informational chunk-size warning. Python test dependencies emit upstream deprecation warnings.
- The one-command launcher was executed, starting the backend on loopback port 8000 and the frontend on 5178. Another application used 5173 and was left alone.

## Browser checks

Codex in-app browser was used directly; no Playwright fallback browser was needed. Checked the native 1536×1024 concept viewport, the initial browser viewport, and mobile 390×844. Mobile document width stayed within the viewport, with no controls overflowing horizontally.

Verified Run→Pause→Step, stimulation, selecting brain regions, motor-neuron silencing, training→completion→Apply→Evaluate, checkpoint listing, the measured-data neural probe, camera controls, and paused replay→Return live. On replay the displayed clock moved to 0.00 s while the backend remained paused at 0.28 s; Return live restored 0.28 s. Export controls construct JSON from recorded simulation/training/probe state. The recording export was invoked, but the in-app browser did not report a completed download event; saved download contents were not independently verified.

Development reloads initially produced a duplicate Three.js warning and transient Vite/WebSocket reconnect messages. Direct Drei module imports and Three.js deduplication remove the duplicate import on a fresh load. WebSocket reconnection preserves the local workbench session; controls are disabled while disconnected. No runtime errors or warnings appeared in the final fresh production page at port 8000. The production build, static mesh routes and live PyTorch connection were verified together.

## Visual fidelity ledger

Reference: design/concept.png. Implementation was inspected in the browser and both the concept and the latest screenshot were inspected with view_image.

| Comparison | Evidence and resolution |
|---|---|
| Composition | Preserved dominant body canvas on the left, neural inspector on the right, and synchronized telemetry beneath. |
| Palette | Preserved charcoal/navy chrome, cool gray arena, lime action controls, blue neural activity and orange selection/muscle accents. |
| Typography | Local DM Sans and IBM Plex Mono; deliberate panel, form and metric scales. Fonts do not require Google Fonts access. |
| Controls | Run, Step, Reset, camera controls, stimulation and silencing remain native interactive controls; no screenshot is used as interface. |
| Brain visualization | Increased point visibility and added central schematic envelope. Region geometry is explicitly schematic and is not an anatomical reconstruction. |
| Body assets | All 69 segments now use the connected anatomical rig and 39 research meshes with shared physical/rendering transforms. The simplified source surfaces intentionally differ from the photoreal design image. |
| Scientific labels | Added visible synthetic-model provenance, physical mm/µN units and six model-population readouts. Removed invented account/settings actions and unsupported anatomical-view tabs. |
| Responsive behavior | Desktop split becomes a vertical body→brain→telemetry flow on mobile. Training and data sections reflow without horizontal overflow. |
| Copy | Differences from the concept are intentional: conditioning and experimental spatial steering are explicitly distinguished from successful learned navigation; measured-data status and simulation limitations replace illustrative claims/values. |

The requested layout and visual system were verified against the concept with the scientific and interactive-geometry deviations above. Pixel identity to the photoreal concept is not claimed.


## Sensory integration

The `arena-senses-v1` sampler and `lif-sensory-histamine-v3` neural model were checked with modality-isolated 100 ms full-graph runs. With odor, touch and proprioception disabled, the all-inputs-off run produced zero spikes/current; vision produced 30,783 modeled spikes, hearing 606, and wind/gravity 5,485. Each active modality also produced nonzero synaptic current. Hearing used a unit-strength 250 Hz source at (2, 1, 1.5) mm; wind used 100 mm/s at 90°. These are input-transmission checks, not evidence of recognition, sensory calibration, or useful behavior. The ignored local report is `data/sensory-validation.json`.

Tests cover pose-dependent eye samples, darkness, deterministic sampling without body mutation, sound distance/frequency sensitivity, modality ablations, annotation-only neural routing, negative histamine transmission, 598-value Gymnasium observations and checkpoint reproduction of neural/body trajectories with sensing enabled. Physical checkpoint tests verify retention of tone settings and sensory version; API tests cover invalid settings, reset/episode changes and dark eye views.

Browser checks at localhost:8000 used 1280×720 and 390×844. The sensory panel displayed both eye images and changing measurements. Enabling hearing/wind, moving the cue, applying/resetting and stepping the measured controller worked; zero illumination blackened both views, and clearing a required numeric field by keyboard disabled Apply. Mobile views and expanded controls had no horizontal overflow. Page identity, rendering, interaction and console checks passed without a framework overlay or fresh errors. The workbench was left paused with vision, hearing and wind enabled. No long-horizon learning or real-animal behavior was validated by these checks.

## Furnished scene validation (2026-09-12)

The regression run passed 46 tests. After final geometry refinements, all 10 scene tests passed, and all 5 sensing tests passed, including the added kitchen neural/body checkpoint replay case. The frontend production build and `git diff --check` passed. Vite retains its existing large Three.js chunk advisory; no build errors occurred.

- Every preset compiles with 69 fly segments, unchanged fly mass and one static geom per scene object. Current object counts, excluding the floor: kitchen 152, living room 152, bedroom 92, garden 468.
- A 0.3 s posture rollout supports the fly at each designated surface with approximately 10 µN total foot load. Raised-surface contacts involve furniture/paving, not the floor geom. No MuJoCo warning counters were raised. Moving the fly beyond the counter edge lets gravity lower it more than 30 mm in 0.1 s; the root is not pinned.
- A downward ray reaches the 900 mm counter at the expected distance. Changing that collider’s reflectance changes retinal samples; head rotation changes the image.
- Kitchen training saves the scene fingerprint, source height, odor and sensory settings. Loading reproduces the posture trajectory; a small annotated neural fixture also reproduces both body positions and neural rates. A mismatched geometry fingerprint rejects loading before changing the live body. These fixtures do not establish whole-MaleCNS navigation.
- Browser review covered all four furnished overviews, object selection/dimensions, kitchen Surface/Fly views, garden click-to-focus, physical stepping, live sensory readouts and the training scene label. At 390 × 844, all five scene choices and camera controls remained usable, with document width equal to viewport width. The viewport override was reset afterward.

The rooms are stylized, static cutaways. Appearance inspection is not evidence of calibrated optics, deformable objects, scene navigation, or flight.
