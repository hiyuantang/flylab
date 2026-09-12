# Body mechanics and control

## Connected anatomy

The model has 69 segments, a free thorax root and 70 hinge degrees of freedom. Each leg has three coxa axes, two trochanter/femur axes, one tibia axis and five tarsal hinges. The first seven degrees of freedom per leg are actuated; the four distal tarsal hinges are passive. Each wing has passive elevation and spread hinges; other body parts have fixed attachments. Wing anchors are fitted 0.25 mm dorsally on the thorax, with an assumed 0.15 rad elevation and 0.6 rad outward spread at rest. Every mesh uses its source origin and the same world transform in MuJoCo and Three.js.

Units are millimeters, grams and seconds, giving force in µN. Source masses plus numerical lower bounds produce a total mass of approximately 1.024 mg. MuJoCo integrates at 0.1 ms with gravity 9810 mm/s², implicitfast integration, mesh-based ground contacts and self-collision among nonadjacent body segments. Wing collision geometry uses 12 smaller convex surface sections per wing, each covering complete source triangles with a 5 µm shell thickness. The rendered source meshes are unchanged. Joint sockets and rigidly connected body parts are excluded from self-collision; distal wing sections additionally collide with the thorax, and all wing sections collide with the head and abdomen.

Before time starts, reset projects joint angles out of self-intersections while preserving joint limits and the root transform. During simulation, MuJoCo contact forces resolve collisions; rendered parts are never moved independently to hide overlaps. Contacts are compliant and may have small transient penetration. The collision regression checks a 15 µm penetration bound during the tested two-second walk and verifies repulsive wing contact forces. Extreme impacts and every possible policy trajectory are not exhaustively validated.

## Effective muscles and walking control

There are 84 antagonistic Hill actuators across 42 active joints. Their fixed moment arms (0.05 mm), maximum force parameter (1400 µN), activation/deactivation constants (2/4 ms), joint limits, damping and springs are modeling choices. These actuators are not an anatomical list of 84 fly muscles. Anatomical tendon routing, calibrated force curves and active wing, neck, mouthpart and abdominal control remain future work.

The current web experiment instead uses the full imported MaleCNS graph, paper-derived neuron timing and direct muscle excitation from mapped motor-neuron rates. It has no gait oscillator or posture tracker. Neural and physical solvers retain 0.1 ms internal steps and exchange sensory/motor values every 20 ms. See [brain fidelity](BRAIN_FIDELITY.md).

### Historical controller benchmarks

The retained reference-test brain outputs 12 motor-group drives. Their per-leg means control gait amplitude and the frequency of six coupled oscillators. Pinned NeuroMechFly joint references supply target angles. A 1 kHz joint-feedback controller converts tracking error into bounded muscle excitations; MuJoCo computes activation, muscle force, joint motion and body movement. This walking scaffold is explicit and is not learned from MaleCNS wiring. Silencing the motor population or VNC removes its excitation; muscle activation then decays.

World-surface contact forces feed back to the synthetic sensory input and VNC. Spatial mode samples a Gaussian odor field at the two moving antennal funiculi. An explicit bearing decoder converts learned approach/avoid preference into asymmetric leg drives. This decoder has access to target bearing; it is not biological odor localization inferred solely from the two antenna readings.

## Physical reinforcement learning

`FlyLab-Locomotion-v0` exposes six leg drives (`synergy`) or 84 direct excitations (`muscle`). Direct muscle control bypasses the walking scaffold. Each action advances 20 ms; the first 338 observations include joint angles/velocities, activations, body orientation/velocity, foot loads, local target, odor, oscillator state and elapsed episode fraction.

Reward equals progress toward the target in mm minus 0.002 times mean squared activation, plus 5 on reaching within 1 mm, or minus 2 on a fall. Episodes terminate on success/fall and truncate at the configured step limit. Reset seeds the target placement. This is an engineering benchmark. The website Physical learning panel runs full-graph neural-gain optimization against actual MuJoCo rollouts. Synthetic odor-decision and posture experiments remain historical test utilities.

## Evidence and limits

Tests verify all parent attachments during motion, bounded upright forward walking for two seconds, pulling muscle forces, ground loads, motor decay, sensory sampling and Gymnasium contracts. Successful learned navigation and general locomotor robustness remain unvalidated. Web physical learning trains MaleCNS population gains; historical posture-feedback tests do not validate neural control. Reports include initial/trained reward, a zero-muscle ablation, and lateral-force/mass-variation checks. The stance task additionally requires maintained body height and joint pose: passive support alone can keep this model upright.

The body is based on a female specimen. The web controller uses the full annotated MaleCNS graph with a partial motor interface. Historical synthetic and posture utilities are not offered as web controllers. Flight, uneven terrain, perturbation recovery and biological fidelity have not been validated. The Python wheel bundles the anatomical meshes for headless use; run the web workbench from the source checkout.


## Configurable mechanics and measured neural interface

Body parameters are immutable and compiled models are cached by configuration, so changing mass, strength or friction cannot mutate another environment. The default joint ranges preserve the tested baseline. The optional reference envelope uses the min/max recorded walking angles for each active axis, includes its neutral angle, and adds 0.35 rad on either side. This bounds a motion reference; it is not a measurement of maximal anatomical rotation. The UI lists all 70 joint ranges, stiffness and damping values.

The full graph interface maps exact tibia/trochanter flexor/extensor and tarsal motor annotations to 28 modeled channels. Leg identity uses subclass and somaSide; mechanical axis and torque sign remain assumptions. Unknown targets receive no invented mapping. Generic contact/proprioceptive and olfactory currents enter annotated sensory neurons. Receptor tuning, visual phototransduction, coxa muscle routing and full-body muscle coverage require further evidence. See the source annotations and [MANC motor-target work](https://elifesciences.org/articles/96084) for anatomical context; cross-specimen mechanical equivalence is not asserted.

Physical training checkpoints store controller parameters, body configuration, physiology, graph fingerprint, environment, neural dynamics version, reward version and evaluations. They keep the anatomical counts fixed. `stand-pose-v2` rewards upright posture, height and joint-pose preservation, with effort/drift penalties. Success requires upright >0.9, drift <0.5 mm, height error <0.3 mm and joint RMS error <0.1 rad at the horizon. Older checkpoints without a reward version used the weaker upright/drift criterion and must not be compared as equivalent successes.

The `lif-step-rate-v2` neural model filters spikes at each integration step with a 50 ms time constant. This makes the decoded firing rate independent of API call duration. Saved evaluations without this dynamics version used batch-averaged rates and require re-evaluation. Legacy policies load with their original fixed trial environment: uniform odor A at 0.7 intensity, target (12, 0) mm. Checkpoint validation completes before changing the live controller; saved environment settings also drive command-line re-evaluation.

## Sensory observations

The headless sensor suite adds 260 channels to Gymnasium (598 total): left then right eye luminance, row-major 8 × 16 each; left/right auditory envelope; left/right wind/gravity proxy. Existing body state remains observable even when a neural sensory channel is disabled. Checkpoints save sensory settings and version alongside the odor environment. Legacy checkpoints without sensory settings disable new vision/hearing/wind inputs. Current neural dynamics are `lif-sensory-histamine-v3`; older evaluations need rerunning. See [SENSING.md](SENSING.md).

## Furnished worlds

Versioned scene definitions supply static furniture and terrain geoms directly to the MuJoCo world. The renderer and head-mounted rays use their shared dimensions and transforms. Reset places the fly on a defined support surface; there is no root-height pin. Counter/table forces contribute to foot feedback. Layouts and limits are documented in [SCENES.md](SCENES.md).

### Named peripheral mechanics

The optional `peripheral-v1` body / `muscle-routing-v4` bridge adds 96 Hill-type channels to the 90-channel pretarsal body. New joints represent proboscis/labellar positioning, internal elastic wall displacement, head yaw, wing feathering and haltere stroke/trim. Fixed-tendon coefficients convert muscle force to joint force; motion is integrated with passive springs, damping and limits. Internal sliders report millimetres, hinge ranges report degrees. Two effective labellar pads share collision geometry with the renderer.

These are executable effective mechanisms, with uncalibrated force capacities, geometry and transmission directions. Shared wing-axis projections do not reconstruct individual sclerite linkages. Fluid transport, asynchronous thorax power and aerodynamic flight remain absent. See [target evidence, assumptions and continuation behavior](NEUROMUSCULAR_MAPPING.md#named-peripheral-mechanisms).
