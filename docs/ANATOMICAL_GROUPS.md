# Anatomical group explorer

The brain panel's **Anatomical groups** view browses the full imported MaleCNS neuron population. Select a superclass, cell type, soma side, and assigned hex column; use **Neurons** to inspect body IDs, predicted transmitters, rates and voltages. Select a neuron to inspect its connection totals. **Connections** reports exact directed pairs and integer synapse counts internally and across the selected boundary, with crossing synapses grouped by the other endpoint's superclass.

## Membership and state

`FullBrain.anatomy` lazily constructs `AnatomyIndex`. Its `partition(group_id)` returns immutable `NeuronGroup` records with read-only NumPy indices into the original neural state. Siblings partition their parent exactly. Missing values form explicit unassigned groups; duplicate column assignments retain every neuron. Source label spelling is preserved. One neuron identity and one dynamic state remain in the global registry.

Grouping does not change the graph, numerical precision, integration schedule, checkpoint format, or physiological model. It adds views and index partitions, not a new GPU execution backend. The simulation continues to use its original full sparse graph, including every cross-group connection.

The hierarchy is an annotation-based navigation tree, not a complete anatomical ontology. Superclasses are not neuropil boundaries; soma side does not establish arbor territory. Hex assignments are partial. Many-to-many neuropil membership and anatomical laminae require synapse-ROI evidence before they can be added. The original six-population schematic and its interventions remain available under **Schematic controls**; explorer selections do not silently change intervention targets.

## API and performance

`GET /api/anatomy` accepts `group` (default `all`), `view` (`groups`, `neurons`, `wiring`), `query`, `offset`, and `limit` (1–100). Returned IDs encode exact annotation paths and resolve independently of navigation history. Responses include breadcrumbs, filtered totals, neural time and graph/annotation SHA256 identities. Unknown group IDs return 404; unavailable measured brains return 503.

Search filters the selected group's subgroups, or neuron IDs/type labels. Pagination only limits returned display rows. Group statistics always include the complete selected membership. Activity is the mean of current per-neuron filtered firing rates; “above 1 Hz” is a display statistic, not a simulation cutoff. Rates refresh sequentially every two seconds, with their simulated timestamp. Live exploration is unavailable during trajectory replay to avoid mixing current and historical state.

Exact wiring summaries read the original integer-count CSR in bounded chunks. Self-connections count once internally. Incoming/outgoing summaries exclude internal edges. A 32-entry per-brain LRU caches structural summaries; firing rates are never cached. Workbench locking keeps a response within a completed simulation state. Large first-time queries can add wall-clock latency without dropping neural steps.

## Validation

`PYTHONPATH=backend uv run pytest tests/test_anatomy.py tests/test_api.py -q`

Tests cover complete/disjoint partitions, unknown annotations, multiple neurons assigned to one column, directed weighted boundary accounting, self-connections, live-state reads, search/pagination, independent group links, invalid requests, and exact neural continuation after inspection. Run `npm run build` from `frontend/` and verify subgroup → neuron → connections navigation and a narrow viewport when editing the UI.
