# Physical environments

FlyLab includes four furnished environments and the original laboratory. Dimensions use millimetres throughout; furniture is human-sized, while the anatomical fly is approximately 2–3 mm long. Scenes are original procedural assets authored in `backend/flylab/scenes.py`, covered by the repository license.

| Scene | Floor extent | Starting surface | Contents |
| --- | --- | --- | --- |
| Kitchen | 3.6 × 3 m | Counter, 900 mm high | Sage cabinetry, tiled backsplash, stove and pan, sink/faucet proxies, refrigerator, breakfast furniture, fruit, mug, bread and crumbs |
| Living room | 4.2 × 3.6 m | Coffee table, 420 mm high | Cushioned sofa, rug, books, botanical prints, television console, lamp and plants |
| Bedroom | 4 × 3.5 m | Bedside table, 520 mm high | Layered bedding, pillows, wardrobe, drawer, reading lamp, books and writing desk |
| Garden | 5 × 4 m | Paving stone, 15 mm high | Raised beds, herbs, planters, bench, watering can, pebbles, grass and food crumbs |

## Navigation and experiments

Choose a scene above the simulation. Selection pauses and resets the episode, moves the odor and virtual sound sources near the starting surface, and preserves controller gains and mechanics. Rooms default to a spatial odor field; the laboratory defaults to a uniform cue. Sensory channels retain their enabled state.

**Overview** shows the cutaway environment and a clickable fly-location marker. **Surface** provides nearby object context; **Fly view** and **Top** inspect the actual body. Orbit and zoom work in every view. Select an object to see its identity and dimensions. Markers and inspectors are observer overlays, not sensory targets.

Physical training captures the current scene, geometry fingerprint, sensory settings, odor field and target. Applying a checkpoint restores that context. A changed layout fingerprint rejects loading before live state changes. Trials currently train stance or local target approach, not a room-navigation curriculum. The separate odor-conditioning bandit has no physical scene.

## Shared geometry and extension

`GET /api/scenes` lists summaries; `GET /api/scenes/{id}` returns the full geometry. Add a builder and identifier in `scenes.py`, then add its choice in `frontend/src/components/World.tsx`. Objects use box half-extents, cylinder radius/half-height, or ellipsoid radii, plus world position and a quaternion in **w, x, y, z** order. Rendering converts that order for Three.js and rotates cylinder axes from Y to Z.

Each object becomes a static MuJoCo world geom. Head-mounted vision rays intersect these same solids. Foot forces include support on furniture and terrain, not only the floor. The fly retains its original mass, joints and muscle model. Headless use: `FlyEnv(scene_id="kitchen", action_mode="posture", task="stand")`.

## Limits

Rooms have two enclosing walls, an open front/side, no ceiling, and finite floors. Furniture, bedding, leaves and coffee surfaces are rigid approximations; doors do not open, objects cannot be pushed, and there is no liquid, fabric or airflow simulation. The sink is an opaque inset proxy. Window panels are opaque colored solids. Vision uses surface luminance rather than rendered shadows, textures, refraction, or fly self-occlusion. Odor remains an assumed horizontal Gaussian field without wall occlusion; sound attenuation is also analytic. No biological navigation or avoidance behavior is established.
