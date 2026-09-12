# Measured optical reference

`eye-directions.json` contains numerical measurements derived from Reiser Lab's [eyemap_T4](https://github.com/reiserlab/eyemap_T4), commit `99d2a43123db636cedb55af9ff31a59657e7d17e`, accompanying Zhao et al., *Eye structure shapes neuron function in Drosophila motion vision*, Nature (2025), DOI `10.1038/s41586-025-09276-5`.

Upstream files: `data/microCT/20240701.RData` and `data/eyemap.RData`. Upstream repository license: GNU GPL v3, retained in `SOURCE-LICENSE.txt`; this attribution and license apply to the derived numerical asset. No upstream R implementation is incorporated into FlyLab's Python code.

Extraction with Python `rdata.read_rda`: convert arrays to NumPy; sort `ucl_rot_sm` by `i_match` into lens order; split by `ind_left_lens` (857 left, 852 right). Copy `lens_ixy` columns 2–3 into right-eye lens order using its one-based first column. Preserve vectors to ten decimal places; normalize at sampling time. No directions are downsampled. Axes are x forward, y left, z dorsal, as defined by the source eye equator. Alignment with the body rig is an assumption.

The hex grid belongs to the optical reference specimen. It is not a measured MaleCNS column-to-retina transformation. FlyLab's centre-aligned transfer of MaleCNS hex coordinates requires validation of axis orientation, equator, correspondence and boundaries before spatial accuracy can be claimed. Left retinal correspondence uses the nearest independently measured left direction to the mirrored right template; it is not a measured left-eye column assignment.
