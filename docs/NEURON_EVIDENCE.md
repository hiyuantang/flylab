# Neuron evidence inventory

Data & model provides separate pages for **Neurons & wiring**, **Sensory inputs**, **Motor outputs**, **Model settings**, and the **Paper reference**, alongside an overview. Every imported neuron has an inspectable evidence record. Search by body ID, type, class, transmitter, side, or nerve; filter by the complete imported superclass list and browse in pages of 30.

## What the labels mean

| Claim | Evidence shown | Interpretation |
| --- | --- | --- |
| Identity and classification | All retained MaleCNS annotation fields | Source annotation. Missing fields remain unknown; this import supplies no overall identity probability. |
| Neuron-to-neuron anatomy | Directed partners and original integer synapse counts | Dataset-supported anatomy from the pinned MaleCNS v1.0 minconf-0.5 release. This is not a claim of complete or error-free biological wiring. |
| Transmitter | Imported `consensus_nt` and original `predicted_nt_confidence` | Source prediction. The numerical score belongs to the source prediction, not overall neuron function or simulation accuracy. Zero remains zero; absent scores remain unknown. |
| Electrical behavior | Current physiology configuration and assumed fast-synaptic sign | Model assumption. Synapse counts do not measure electrical efficacy. A zero sign retains anatomy but supplies no fast synaptic current. |
| Behavioral role | Explicit assessment status | Not independently assessed for each neuron. Source class/type labels do not constitute individual functional validation. |
| Muscle interface | Existing per-motor target evidence, source scores and mechanical interpretation | Target identity and simulated force path have separate confidence descriptions. |
| Sensory interface | Implemented input groups, selected profiles, modality switches and route limitations | Annotation-based recipients with assumed transduction. Compound visual columns are inferred from connectivity; optical registration across specimens is unvalidated. |

[MaleCNS documentation](https://male-cns.janelia.org/download/) distinguishes curated annotations, aggregate neurotransmitter predictions, and connectivity tables. The imported aggregate connection table does not retain individual synapse confidence values. Its minimum-confidence filename must not be interpreted as a neuron-level or pair-level confidence score.

Sensory records expose both legacy and alternative routes where applicable. A selected profile and enabled modality do not imply a nonzero stimulus. Unsupported compound visual routes include their specific unresolved reason. FeCO angle/velocity tuning remains an inference with uncalibrated response curves. Neurons without external input still retain their imported neural connections. Experimental current pulses and silencing are described separately from anatomical wiring.

## Coverage and boundaries

The installed graph contains **166,700 neurons**, **25,582,938 directed pairs**, and **124,177,617 retained synapses**. All imported neurons receive field-based explanations, including intrinsic, projection, descending, ascending, sensory, motor, and other annotated classes. These descriptions are generated from source fields and implemented routing rules; they are not 166,700 independent literature reviews.

The sensory page includes 17,937 neurons: all imported superclasses containing `sensory`, plus recipients in implemented sensory input groups. This is an inventory definition, not a count of validated or currently active sensors. All 815 motor neurons retain the more detailed motor evidence inventory.

There are 2,194 imported neurons without a type label, 178 without a transmitter label, and 1,035 without a transmitter prediction score. The existing importer excludes 44,877 unclassified source annotation rows and reports crossing connections. This evidence feature does not change that selection or modify any graph, neural state, or physical routing.

## Inspection and export

- `GET /api/evidence`: population counts, missingness, source manifest and confidence policy.
- `GET /api/evidence/neurons`: paginated list; `query`, `superclass`, `scope=all|sensory|motor`, `offset`, and `limit` (maximum 100).
- `GET /api/evidence/neurons/{body_id}`: complete evidence, annotations, sources, current physiology and peripheral interfaces.
- Existing `GET /api/neural-view/neuron/{body_id}`: paginated incoming or outgoing anatomical partners and counts.

The UI exports the current list page or one complete neuron record. Sources retain the imported file URLs and SHA256 values. Detail responses describe configuration at request time; reopen a record after changing settings to refresh it.

`PYTHONPATH=backend .venv/bin/python scripts/audit_neuron_evidence.py` checks every retained annotation field against the original source table and both imported transmitter fields against their source table. It generates all evidence records into ignored `data/neuron-evidence.jsonl.gz` and a compact [validation report](results/neuron-evidence.json). The export uses an explicitly configured audit profile (compound vision and FeCO feedback with default modality switches); it is not a snapshot of live stimulus settings. It contains generated explanations for every imported neuron without copying bulk data into Git.

The full installed-data audit passed for all 166,700 records. Focused tests cover exhaustive pagination on a fixture, search/filter boundaries, null and zero prediction scores, unresolved visual inputs, profile changes, API validation, and read-only neural state. Browser checks cover desktop/mobile page navigation, search, pagination, individual evidence export, incoming/outgoing wiring, visual/FeCO details, and motor evidence.
