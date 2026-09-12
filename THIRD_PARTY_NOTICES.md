# Third-party notices

The Apache-2.0 license at the repository root applies to FlyLab's original code and documentation. It does not replace the licenses of the materials below or of installed dependencies.

## NeuroMechFly / FlyGym anatomy and walking references

- Copyright 2023–2026 The NeuroMechFly v2 Authors.
- Source: [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), commit `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`.
- Source directory: `src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/`.
- Included meshes: 39 unchanged STL files, listed with source URLs, sizes, Git blob IDs and SHA256 hashes in [sources.json](frontend/public/models/sources.json).
- Derived backend assets: `anatomy.json` contains segment frames, masses, hierarchy and neutral angles from the upstream rig; `walking_reference.npz` contains numeric joint references converted from `src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl`. Their provenance is stored alongside the assets.
- License: Apache-2.0. The complete upstream copyright and license text is retained with both the [meshes](frontend/public/models/FLYGYM-LICENSE.txt) and [backend assets](backend/flylab/assets/FLYGYM-LICENSE.txt).

At runtime FlyLab uniformly converts mesh units to millimeters, mirrors left-side meshes for right-side segments, and applies shared anatomical transforms in physics and rendering. Colors, transparency and decorative bristles are added. Gait conversion changes right yaw/roll signs to match symmetric joint axes and uses periodic linear interpolation. Wing collision sections are generated from source triangles; wing anchors/resting angles and reset joint angles are adjusted for clearance. Muscle transmissions, strengths, limits and the feedback controller are FlyLab assumptions. The source body geometry derives from an adult female fly; it is not a measured male body or a neural-to-muscle mapping. See [asset provenance](docs/ASSETS.md).

## MaleCNS dataset

- Dataset: Male CNS connectome, v1.0.
- Attribution: FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research.
- Project: <https://male-cns.janelia.org/>.
- Downloads and license statement: <https://male-cns.janelia.org/download/>.
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The bulk dataset is downloaded separately and is excluded from this repository. The importer uses body annotations, neurotransmitter predictions, and connection weights. It creates a selected, induced subgraph, retains original integer synapse counts, and records omitted boundary connections and file hashes in `data/manifest.json`. The runtime neural probe additionally derives normalized model efficacies and assumes transmitter signs and LIF dynamics. Those modeling choices are FlyLab assumptions, not experimentally measured physiology supplied by the dataset authors.

When redistributing data or extracted subsets, retain appropriate dataset attribution, source and license links, and an indication of changes. Do not imply endorsement by the dataset creators. FlyLab's code license does not relicense the dataset.

## Fonts and icons

The frontend obtains these assets through npm. Installed packages and built font files are excluded from Git. Copies of their complete license texts are preserved in `frontend/public/assets/licenses/`, and Vite includes that directory in production builds. These notices are served at `/assets/licenses/` by the production backend.

| Component | Attribution | License |
|---|---|---|
| [DM Sans](https://github.com/googlefonts/dm-fonts), via `@fontsource/dm-sans` 5.3.0 | Copyright 2014 The DM Sans Project Authors | [SIL OFL-1.1](frontend/public/assets/licenses/DM-Sans-OFL.txt) |
| [IBM Plex Mono](https://github.com/IBM/plex), via `@fontsource/ibm-plex-mono` 5.3.0 | Copyright 2017 IBM Corp. | [SIL OFL-1.1](frontend/public/assets/licenses/IBM-Plex-Mono-OFL.txt) |
| [Lucide React](https://github.com/lucide-icons/lucide) 0.468.0 | Lucide Contributors; includes portions from Cole Bemis's Feather icons | [Upstream ISC notice, including Feather attribution](frontend/public/assets/licenses/Lucide-LICENSE.txt) |

## Software dependencies

Python and npm dependencies are installed separately under their own licenses. `uv.lock` and `frontend/package-lock.json` record the resolved versions. PyTorch, MuJoCo, FastAPI, NumPy, PyArrow, React, Three.js, React Three Fiber, and Drei are dependencies, not original FlyLab code. Vite emits an additional `frontend/dist/assets/THIRD_PARTY_LICENSES.md` containing the bundled JavaScript dependencies’ license texts at build time. Preserve applicable upstream copyright, license, and notice files when redistributing dependencies or application bundles. The notices above identify the assets directly included in this source repository and the fonts and icons used by its interface; they are not an exhaustive inventory of transitive software dependencies.

## Shiu computational brain reference

Equations, experiment parameters and receptor ID lists are adapted from [Philip Shiu and Nico Spiller's Drosophila brain model](https://github.com/philshiu/Drosophila_brain_model), commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`, accompanying [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9). Copyright 2023 Philip Shiu and Nico Spiller. The [complete upstream MIT license](backend/flylab/assets/SHIU-MIT-LICENSE.txt) is included. FlyLab implements the equations in PyTorch, adds explicit scheduling and continuation tests, and separately applies them to MaleCNS. The paper's full FlyWire female v630 tables are downloaded separately, checksum-pinned, and excluded from Git; their provenance is included in experiment results. This model is distinct from the MaleCNS specimen and is not an endorsement or a replication of every published result.

## Compound-eye numerical reference

The derived numerical asset `backend/flylab/assets/retina/eye-directions.json` uses Reiser Lab's [eyemap_T4](https://github.com/reiserlab/eyemap_T4), commit `99d2a43123db636cedb55af9ff31a59657e7d17e`, accompanying Zhao et al., Nature (2025), DOI `10.1038/s41586-025-09276-5`. The upstream GPL v3 license is retained in `backend/flylab/assets/retina/SOURCE-LICENSE.txt`. See that directory's README for source files, extraction, attribution and registration limits. This asset's license is distinct from FlyLab's own code license.
