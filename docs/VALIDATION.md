# Validation record

Current automated result: **17 tests passed**; TypeScript and production build passed.

Verified locally on 2026-09-12 with Python 3.12, PyTorch 2.14, MuJoCo 3.13, React 19.2 and Three.js 0.180. Dependency versions are locked in uv.lock and frontend/package-lock.json.

At the initial repository setup, the source was also checked against an export of the staged source with no bulk dataset or existing frontend dependencies: backend tests reported **9 passed, 1 skipped** (the optional measured-data integrity test), and a fresh `npm ci` followed by the production build passed. Python source and wheel distributions build successfully and include the Apache-2.0 license metadata, LICENSE, and NOTICE. Frontend builds retain the font/icon license files and emit bundled dependency license texts at `assets/THIRD_PARTY_LICENSES.md`.

## Articulated body checks

The expanded body was checked at `http://127.0.0.1:8000/` in the Codex in-app browser at desktop 1536×1024, mobile 390×844 and the default panel size. Run/pause, perspective/top views, camera reset, skeleton display, eye selection (showing its head attachment), environment switching and changing live muscle/contact values worked. Mobile content remained within the viewport. The production build and attributed assets loaded successfully. Earlier server/frontend version mismatches produced errors during development; the final matching build did not reproduce them.

New automated checks cover all 69 parent attachments during motion, 70 hinge degrees of freedom, 84 muscle actuators, two-second upright forward walking, pulling muscle forces, ground loads, decay after silencing, moving antennal odor sampling, environment reset semantics and Gymnasium action/observation/seed/time-limit contracts. The current observation contains 338 values. Collision tests check a separated reset pose, wing contact response and sampled walking penetration below 0.015 mm. All 39 mesh hashes were independently matched to the pinned upstream Git tree. Before the wing-joint update, a wheel built from the source distribution was extracted into an isolated temporary directory; it included all 39 meshes and their license, instantiated the body, and returned a valid observation after a physical step.

Successful navigation and general locomotion robustness remain unvalidated. Direct muscle control is exposed through Gymnasium; the web training panel still trains the odor decision task. See [PHYSICS.md](PHYSICS.md) for assumptions and limitations.

The collision update was also inspected in desktop and mobile views, including Run/Pause and perspective/top/side camera controls. Wings remain above the abdomen in the inspected side view. The favicon uses the title's bug icon and is served successfully by both local servers. No fresh browser errors appeared during the final checks.

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
