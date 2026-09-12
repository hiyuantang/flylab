# MaleCNS anatomical architecture and fidelity audit

## Scope and reference

This audit evaluates the local MaleCNS v1.0 data and FlyLab's imported graph. It does not change the simulation. Reproduction code is in `scripts/audit_malecns.py`; recorded aggregates, timestamps, source URLs and SHA256 hashes are in `docs/results/malecns-architecture-audit.json`. The companion notebook is `docs/notebooks/malecns_architecture_audit.ipynb`.

The reference population is **every annotation with a non-null superclass**, including tentative classes. This is an explicit identified-neuron selection, not a claim to include every biological neuron. All connections induced by those IDs are retained without a weight threshold. Connection means one directed neuron pair; synapse count is its integer weight.

Sources: [MaleCNS downloads](https://male-cns.janelia.org/download/) and [authors' supplementary data, pinned revision](https://github.com/flyconnectome/2025malecns/tree/67767d2233657983993ff6c2be48e836a935863c). The latter includes traversal ranks, DN/AN clusters, column assignments, and regional synapse-capture estimates. Downloaded supplements remain in ignored `data/anatomy-reference/`; their provenance is copied into the result.

## Measured inventory

| Measurement | Result |
|---|---:|
| Annotation rows | 211,577 |
| Selected neuron IDs | 166,700 |
| Directed connections in imported graph | 25,582,938 |
| Synapses represented by those connections | 124,177,617 |
| Literal non-null type labels in selected population | 11,751 |
| Selected neurons without a type label | 2,194 |
| Selected neurons with both optic-column hex coordinates | 23,720 |
| Superclass labels | 27 |

Type labels can be composite. Their number is not proof of 11,751 distinct physiological cell types. Superclasses describe broad populations, not exclusive anatomical neuropils. Neither label should automatically determine a neural-network layer.

The hex-coordinate population spans 15 labels: L1, L2, L3, L5, C2, C3, T1, Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9 and Tm20. There are **31 duplicate side/type/coordinate groups**. This check used `somaSide`, which itself needs anatomical validation as a column-side key. A regular one-cell-per-type-per-column grid cannot be assumed from these fields. Missing column coordinates are not absent neurons.

The DN/AN supplement joins **3,160 unique selected neurons**; **320** have nonempty function annotations. These are literature-linked annotations, not an experimentally validated mapping from every neuron to a body actuator. The downloaded column workbook and regional capture table are available for follow-up; this audit does not yet use them to assign synapses to anatomical layers.

## Wiring establishes recurrence

**7,647,520 directed connections have a reverse connection**, excluding self-connections: 3,823,760 reciprocal neuron pairs. For example, body 10003 → 10011 has 6 synapses, while 10011 → 10003 has 24. There are also 101 self-connections. These measured cycles establish recurrent wiring independently of any choice of PyTorch layer.

The published sensory traversal ranks join 166,335 selected neurons. Comparing the median rank of the two endpoints gives:

| Direction relative to median traversal rank | Connections | Synapses |
|---|---:|---:|
| Increasing rank | 8,033,503 | 42,623,322 |
| Same rank | 12,501,804 | 59,901,012 |
| Decreasing rank | 5,046,987 | 21,652,035 |
| Missing endpoint rank | 644 | 1,248 |

These ranks summarize repeated graph traversal from sensory seeds. They are **not histological layers, transmission delays, or a feedforward topological ordering**. Same-rank connections need not form cycles individually. Reciprocal pairs provide direct cycle evidence.

The result JSON includes the complete 27 × 27 superclass connection matrix, storing nonzero cells. For example, optic-lobe intrinsic neurons have 8,799,556 within-class connections and 2,067,244 outgoing connections to other classes. Descending neurons have 75,273 within-class connections and 542,455 outgoing connections to other classes. Thus internal connectivity does not dominate for every kind of population. These are counts, not connection densities or synaptic-strength comparisons; different population sizes affect them. Within-class counts include self-connections.

## Selection boundary is a major unresolved issue

The raw file contains 151,856,684 segment-pair rows and 311,833,243 synapses. Those are not all identified neuron-to-neuron connections.

The 44,877 excluded annotation rows include 11,864 labeled `Glia`, 12,071 `Orphan`, 10,751 `Unimportant`, and other statuses. These are literal curator labels, not permission to delete biological elements or evidence that each row is a separate missing neuron.

Across the selection boundary there are 117,436,340 connection rows and 177,167,703 synapses. In particular:

| Direction | Connection rows | Synapses |
|---|---:|---:|
| Included neuron → unannotated segment | 112,452,244 | 170,414,577 |
| Unannotated segment → included neuron | 4,552,783 | 5,614,376 |

Unannotated segments must be investigated with segmentation, reconstruction status and synapse-location/capture evidence. They cannot be counted as complete neurons, silently discarded as outliers, or used to calculate a biological completeness percentage. This audit classifies their boundary totals; it does not resolve their identities.

The authors' downloaded counting notebook uses v0.9, excludes tentative superclasses, and also reports connections after a ≥5-synapse threshold. FlyLab uses v1.0, includes tentative classes, and applies no connection-weight threshold. Published headline counts therefore need matched versions and selection rules before comparison.

## Architecture decision

Use one state record per measured neuron and preserve individual directed connection weights. Organize storage and execution by anatomical metadata; preserve all cross-block routes. A neuron may arborize across multiple neuropils, so anatomical membership may be many-to-many while its simulated identity remains unique.

```text
MaleCNS neuron IDs + types + columns + individual synapse counts
                              |
                 neuron registry and sparse graph
                              |
        +---------------------+----------------------+
        |                     |                      |
  visual type/column    central-brain groups      VNC groups
        |<------ measured cross-group wiring ------->|
        +------ local and cross-group feedback ------+
                              |
           per-neuron dynamics advanced through time
                              |
             sensory/body interfaces with provenance
```

This is a sparse dynamical graph, implementable with PyTorch tensors and sparse/event kernels. It is not a stack of independent generic RNNs. Anatomical layers can guide indexing and analysis while retaining feedback across layers. A CNN-style implementation requires evidence for its spatial pattern and weight-sharing assumptions; it cannot replace individual edges merely because a region processes vision.

Next structural work: use synapse ROI coordinates to resolve regional/layer membership, inspect ambiguous column assignments, build type/column-specific connection statistics, and validate a block-partitioned graph against the original IDs and weighted edges. Keep cell-type physiological evidence separate from anatomical labels. Until validated, global LIF parameters, neurotransmitter effects, sensory encoding and muscle coupling remain model assumptions.

## Defining the proposed 99% target

Individual variation motivates comparing homologous types and circuits across specimens. It does not establish that any arbitrary 1% deletion is biologically harmless. Use the measured specimen as the reproducible structural reference first.

Track separate metrics rather than one overall fidelity score:

- **Neuron retention:** reference neuron IDs retained / reference IDs; also report unexpected additions and identity mapping errors.
- **Directed-edge precision and recall:** compare exact ordered endpoint pairs, so extra edges cannot compensate for missing edges.
- **Synapse-count fidelity:** report absolute count errors on the union of edges and retained reference synapse mass. Equal global sums can conceal rewiring.
- **Anatomical stratification:** repeat retention and count checks per type, side and validated region; report small populations explicitly. A high global percentage can conceal the loss of an entire rare circuit.
- **Physiology and behavior:** separate intervention, spike-time/rate and behavioral tests with uncertainty. Structural metrics do not imply equivalent dynamics or digital life.

Keep exact imported wiring as the default; treat 99% as a proposed minimum structural target, not a deletion allowance or a current biological-fidelity claim. Any exception needs affected IDs/edges, evidence, reason, before/after metrics and a reversible mapping. Rarity alone is insufficient. No exceptions or neural pruning were applied in this audit.

## Validation and reproduction

Source and supplement hashes match their manifests; graph hashes match the import manifest. Imported IDs exactly match the declared selection. CSR indices are valid, sorted and unique; counts are positive integers. A full streaming source scan reproduces the retained connection/synapse totals and raw-source totals. This audit is not a new edge-by-edge reimport comparison and does not validate source reconstruction accuracy.

```bash
PYTHONPATH=backend uv run python scripts/audit_malecns.py
PYTHONPATH=backend uv run pytest tests/test_dataset_audit.py -q
```

Three focused tests pass: unknown segment handling, traversal direction including missing ranks, and integer synapse aggregation beyond float64's exact-integer range. Raw files, graph arrays, saved simulation state and runtime equations remain unchanged.
