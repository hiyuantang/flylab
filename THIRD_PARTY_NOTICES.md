# Third-party notices

The Apache-2.0 license at the repository root applies to FlyLab's original code and documentation. It does not replace the licenses of the materials below or of installed dependencies.

## NeuroMechFly / FlyGym visual meshes

- Copyright 2023–2026 The NeuroMechFly v2 Authors.
- Source: [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), commit `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`.
- Source directory: `src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/`.
- Included files in `frontend/public/models/`: `c_head.stl`, `c_thorax.stl`, `c_abdomen12.stl`, `c_abdomen3.stl`, `c_abdomen4.stl`, `c_abdomen5.stl`, `c_abdomen6.stl`, `l_eye.stl`, and `l_wing.stl`.
- License: Apache-2.0. The complete upstream copyright and license text is retained in [FLYGYM-LICENSE.txt](frontend/public/models/FLYGYM-LICENSE.txt).

The STL files are unchanged. At runtime FlyLab centers, repositions, scales, and recolors their geometry, and mirrors an eye. Legs and the physical rig are schematic. The source body geometry derives from an adult female fly; it is not a measured male body or a neural-to-muscle mapping. See [asset provenance](docs/ASSETS.md).

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
