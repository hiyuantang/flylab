# Body mechanics and control

## Connected anatomy

The model has 69 segments, a free thorax root and 70 hinge degrees of freedom. Each leg has three coxa axes, two trochanter/femur axes, one tibia axis and five tarsal hinges. The first seven degrees of freedom per leg are actuated; the four distal tarsal hinges are passive. Each wing has passive elevation and spread hinges; other body parts have fixed attachments. Wing anchors are fitted 0.25 mm dorsally on the thorax, with an assumed 0.15 rad elevation and 0.6 rad outward spread at rest. Every mesh uses its source origin and the same world transform in MuJoCo and Three.js.

Units are millimeters, grams and seconds, giving force in µN. Source masses plus numerical lower bounds produce a total mass of approximately 1.024 mg. MuJoCo integrates at 0.1 ms with gravity 9810 mm/s², implicitfast integration, mesh-based ground contacts and self-collision among nonadjacent body segments. Wing collision geometry uses 12 smaller convex surface sections per wing, each covering complete source triangles with a 5 µm shell thickness. The rendered source meshes are unchanged. Joint sockets and rigidly connected body parts are excluded from self-collision; distal wing sections additionally collide with the thorax, and all wing sections collide with the head and abdomen.

Before time starts, reset projects joint angles out of self-intersections while preserving joint limits and the root transform. During simulation, MuJoCo contact forces resolve collisions; rendered parts are never moved independently to hide overlaps. Contacts are compliant and may have small transient penetration. The collision regression checks a 15 µm penetration bound during the tested two-second walk and verifies repulsive wing contact forces. Extreme impacts and every possible policy trajectory are not exhaustively validated.

## Effective muscles and walking control

There are 84 antagonistic Hill actuators across 42 active joints. Their fixed moment arms (0.05 mm), maximum force parameter (1400 µN), activation/deactivation constants (2/4 ms), joint limits, damping and springs are modeling choices. These actuators are not an anatomical list of 84 fly muscles. Anatomical tendon routing, calibrated force curves and active wing, neck, mouthpart and abdominal control remain future work.

The synthetic brain outputs 12 motor-group drives. Their per-leg means control gait amplitude and the frequency of six coupled oscillators. Pinned NeuroMechFly joint references supply target angles. A 1 kHz joint-feedback controller converts tracking error into bounded muscle excitations; MuJoCo computes activation, muscle force, joint motion and body movement. This walking scaffold is explicit and is not learned from MaleCNS wiring. Silencing the motor population or VNC removes its excitation; muscle activation then decays.

Ground contact forces feed back to the synthetic sensory input and VNC. Spatial mode samples a Gaussian odor field at the two moving antennal funiculi. An explicit bearing decoder converts learned approach/avoid preference into asymmetric leg drives. This decoder has access to target bearing; it is not biological odor localization inferred solely from the two antenna readings.

## Physical reinforcement learning

`FlyLab-Locomotion-v0` exposes six leg drives (`synergy`) or 84 direct excitations (`muscle`). Direct muscle control bypasses the walking scaffold. Each action advances 20 ms; the 338 observations include joint angles/velocities, activations, body orientation/velocity, foot loads, local target, odor, oscillator state and elapsed episode fraction.

Reward equals progress toward the target in mm minus 0.002 times mean squared activation, plus 5 on reaching within 1 mm, or minus 2 on a fall. Episodes terminate on success/fall and truncate at the configured step limit. Reset seeds the target placement. This is an engineering benchmark; the website's existing training panel still trains only the odor decision task.

## Evidence and limits

Tests verify all parent attachments during motion, bounded upright forward walking for two seconds, pulling muscle forces, ground loads, motor decay, sensory sampling and Gymnasium contracts. Successful learned navigation and general locomotor robustness remain unvalidated.

The body is based on a female specimen, the embodied neural circuit is synthetic, and the measured MaleCNS graph remains a separate probe. Flight, uneven terrain, perturbation recovery and biological fidelity have not been validated. The Python wheel bundles the anatomical meshes for headless use; run the web workbench from the source checkout.
