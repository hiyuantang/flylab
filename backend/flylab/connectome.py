"""Versioned MaleCNS induced-subgraph import and an independent sparse LIF probe.

Preserves measured counts separately from assumed efficacy. Missing boundaries
are counted, never silently treated as evidence of disconnected biology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch

SOURCE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {"body-annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather", "connectome-weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5.feather", "body-neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather"}


def batches(path):
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def import_graph(directory: Path, seed=10001, limit=256):
    annotations = feather.read_table(directory/"body-annotations.feather").to_pylist()
    eligible = {int(r["bodyId"]): r for r in annotations if r["superclass"] is not None}
    if seed not in eligible:
        raise ValueError("Seed is not an annotated neuron with a superclass")
    neighbors: Counter = Counter()
    rows = 0
    for batch in batches(directory/"connectome-weights.feather"):
        rows += batch.num_rows
        pre, post = batch.column("body_pre"), batch.column("body_post")
        selected = batch.filter(pc.or_(pc.equal(pre, seed), pc.equal(post, seed)))
        for r in selected.to_pylist():
            other = int(r["body_post"] if r["body_pre"] == seed else r["body_pre"])
            if other in eligible and other != seed:
                neighbors[other] += r["weight"]
    ids = [seed] + [n for n, _ in sorted(neighbors.items(), key=lambda x: (-x[1], x[0]))[:limit-1]]
    lookup = {body: i for i, body in enumerate(ids)}
    values = pa.array(ids, type=pa.uint64())
    edge_pre, edge_post, weights = [], [], []
    boundary_rows = boundary_synapses = 0
    for batch in batches(directory/"connectome-weights.feather"):
        ins = pc.is_in(batch.column("body_pre"), value_set=values)
        outs = pc.is_in(batch.column("body_post"), value_set=values)
        boundary = batch.filter(pc.xor(ins, outs))
        boundary_rows += boundary.num_rows
        boundary_synapses += pc.sum(boundary.column("weight")).as_py() or 0
        for r in batch.filter(pc.and_(ins, outs)).to_pylist():
            edge_pre.append(lookup[int(r["body_pre"])]); edge_post.append(lookup[int(r["body_post"])])
            weights.append(int(r["weight"]))
    nt_table = feather.read_table(directory/"body-neurotransmitters.feather")
    nts = {int(r["body"]): r for r in nt_table.filter(pc.is_in(nt_table["body"], value_set=values)).to_pylist()}
    neurons = [{"body_id": i, "type": eligible[i]["type"], "class": eligible[i]["superclass"], "side": eligible[i]["somaSide"], "position": eligible[i]["somaLocation"], "transmitter": nts.get(i, {}).get("consensus_nt"), "transmitter_confidence": nts.get(i, {}).get("predicted_nt_confidence")} for i in ids]
    np.savez_compressed(directory/"probe-graph.npz", body_ids=np.array(ids, dtype=np.int64), pre=np.array(edge_pre, dtype=np.int64), post=np.array(edge_post, dtype=np.int64), counts=np.array(weights, dtype=np.int64))
    sources = []
    for local, remote in FILES.items():
        path = directory/local
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append({"file": local, "url": SOURCE+remote, "bytes": path.stat().st_size, "sha256": digest})
    manifest = {"dataset": "male-cns:v1.0", "license": "CC-BY", "source": "https://male-cns.janelia.org/download/", "annotation_rows": len(annotations), "eligible_annotation_rows": len(eligible), "source_connection_rows": rows, "neurons": neurons, "edge_count": len(weights), "synapse_count": sum(weights), "seed": seed, "boundary_connection_rows": boundary_rows, "boundary_synapse_count": boundary_synapses, "selection": f"Seed {seed} and up to {limit-1} directly connected annotated neighbors ranked by summed incoming+outgoing synapse count. Eligibility: non-null superclass. All edges induced by selected IDs are retained. Boundary inputs are absent from probe dynamics.", "sources": sources, "assumptions": "LIF physiology is assumed. Counts are scaled to efficacy; only predicted GABA is treated as inhibitory. Other transmitters are treated as excitatory in this deliberately simple probe; receptor-specific effects are unresolved. No link to body muscles is inferred."}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class ConnectomeProbe:
    def __init__(self, directory):
        self.manifest = json.loads((directory/"manifest.json").read_text())
        data = np.load(directory/"probe-graph.npz", allow_pickle=False)
        self.ids = data["body_ids"].tolist()
        n = len(self.ids)
        signs = torch.tensor([-1. if r["transmitter"] == "gaba" else 1. for r in self.manifest["neurons"]])
        pre, post = torch.tensor(data["pre"]), torch.tensor(data["post"])
        counts = torch.tensor(data["counts"], dtype=torch.float32)
        # Normalize the efficacy by postsynaptic anatomical input count. Raw counts
        # remain untouched in probe-graph.npz. This is a model assumption.
        total = torch.zeros(n).scatter_add_(0, post, counts).clamp_min(1)
        values = signs[pre]*counts/total[post]*1.5
        self.weights = torch.sparse_coo_tensor(torch.stack((post, pre)), values, (n, n), check_invariants=True).coalesce()

    @torch.no_grad()
    def pulse(self, body_id, amplitude=2., duration_ms=100, steps=500):
        if body_id not in self.ids:
            raise ValueError("Neuron is not in the loaded subgraph")
        index = self.ids.index(body_id)
        n = len(self.ids)
        voltage, refractory, current, spike_counts = [torch.zeros(n) for _ in range(4)]
        spikes = torch.zeros(n)
        history = []
        for t in range(steps):
            current = current*.82 + torch.sparse.mm(self.weights, spikes[:, None]).squeeze(1)
            drive = current.clone()
            if t < duration_ms:
                drive[index] += amplitude
            refractory = (refractory-1).clamp_min(0)
            voltage += (-voltage + drive)*.05
            voltage[refractory > 0] = 0
            spikes = ((voltage >= 1.) & (refractory == 0)).float()
            voltage[spikes.bool()] = 0
            refractory[spikes.bool()] = 2
            spike_counts += spikes
            if t % 5 == 0:
                history.append({"time_ms": t, "voltage": float(voltage[index]), "population_spikes": int(spikes.sum())})
        active = [{"body_id": self.ids[i], "type": self.manifest["neurons"][i]["type"], "spikes": int(count)} for i, count in enumerate(spike_counts) if count > 0]
        return {"stimulated": body_id, "duration_ms": steps, "active_neurons": len(active), "spikes": int(spike_counts.sum()), "neurons": sorted(active, key=lambda r: -r["spikes"]), "history": history, "physiology": "Assumed LIF parameters; measured connectivity; isolated subgraph"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    result = import_graph(args.data, args.seed, args.limit)
    print(json.dumps({k: result[k] for k in ["dataset", "edge_count", "synapse_count", "boundary_connection_rows"]}))
