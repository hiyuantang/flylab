# Assets and scientific interpretation

## Anatomical body

The 39 STL files in `frontend/public/models/` come from [NeLy-EPFL/FlyGym](https://github.com/NeLy-EPFL/flygym) at commit `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`, under `src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/`. All are unchanged; [sources.json](../frontend/public/models/sources.json) records source URLs, byte counts, Git blob IDs and SHA256 hashes. The upstream Apache-2.0 license is retained alongside them.

NeuroMechFly geometry derives from micro-CT imaging of an adult female fly. It does not establish male anatomy or a neural-to-muscle mapping. The renderer and MuJoCo use the same anatomical frames and uniformly convert meters to millimeters. Left meshes are mirrored for right-side anatomy. Segments are neither centered independently nor resized to fit a schematic rig.

The 69 connected parts include the head, mouthparts, antennae, eyes, thorax, abdominal segments, wings, halteres and eight segments per leg. Leg joints and two passive hinges per wing move physically. Wing anchors and resting angles are fitted to clear the thorax and abdomen; see [PHYSICS.md](PHYSICS.md) for the explicit offsets. Other attachments remain fixed. Runtime colors, translucent wings and decorative bristles improve appearance but are not measured surface properties. MuJoCo uses convex hulls for body contacts and smaller convex surface sections for the curved wings, not exact nonconvex triangle collisions.

## Derived parameters and gait

`backend/flylab/assets/anatomy.json` preserves source rigging transforms, segment masses and neutral angles with an explicit parent hierarchy. The root is translated to the local arena origin. Numerical lower bounds on mass/inertia and the chosen joint configuration are described in [PHYSICS.md](PHYSICS.md).

`walking_reference.npz` contains numeric single-step joint references from the same pinned repository. `walking_reference.json` records the original path and SHA256. Conversion selects six legs and seven active angles per leg, flips right yaw/roll signs for symmetric axes, and uses periodic linear interpolation. Runtime loading disables pickle. The references set targets for finite-force muscles; they do not set rendered poses directly. Both derived assets retain a copy of the upstream license.

## Other visuals

The muscle overlay shows modeled leg activity, not reconstructed muscle volumes. The brain view uses schematic point clouds rather than measured soma locations or region boundaries; values and stimulation highlights come from the synthetic circuit.

`docs/design/concept.png` is an Image Gen design reference with its prompt in `docs/design/prompt.txt`. It is not used as interactive UI. Font and icon attribution is in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
