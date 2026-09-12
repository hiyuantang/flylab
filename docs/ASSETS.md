# Visual assets and scientific interpretation

The nine STL files in frontend/public/models were obtained from NeLy-EPFL/FlyGym at commit 38c8ec61034cd59bc5ba0de20688d4a3c0000d60, under src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces. Original files are retained unchanged. The Apache 2.0 license is included alongside them.

Source repository: https://github.com/NeLy-EPFL/flygym

Source model: NeuroMechFly, derived from micro-CT imaging of an adult female fly. These assets are visual geometry, not evidence of a male specimen's anatomy or neural-to-muscle mapping. The renderer centers and independently scales the meshes to fit the schematic MuJoCo rig, mirrors an eye, recolors segments, and uses simplified capsules for legs. Muscle overlays represent modeled activation, not reconstructed muscle volumes. The body physics still uses simplified collision shapes.

The brain view is a selectable schematic assembled from point clouds. It does not depict measured soma positions or anatomical region boundaries. Values and stimulation highlights come from the live reference circuit.

The concept screenshot in docs/design/concept.png was made using the built-in Image Gen tool. Its exact prompt is saved in docs/design/prompt.txt. It is a design reference only; no screenshot is used as interactive UI.
