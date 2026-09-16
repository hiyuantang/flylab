import { useEffect, useState } from "react";
import { downloadJSON, request } from "../lib/api";
import type {
  EvidenceDetail,
  EvidenceMetadata,
  EvidenceNeuron,
  MotorMappingRecord,
} from "../lib/types";
import { MotorEvidence } from "./MotorEvidence";
import Select from "./Select";
import "./NeuronEvidence.css";

type Page = {
  total: number;
  offset: number;
  limit: number;
  records: EvidenceNeuron[];
};
type Wiring = {
  total: number;
  incoming_total: number;
  outgoing_total: number;
  incoming_synapses: number;
  outgoing_synapses: number;
  partners: { body_id: number; type: string | null; synapses: number }[];
};

export function NeuronEvidence({ sensory = false }: { sensory?: boolean }) {
  const [meta, setMeta] = useState<EvidenceMetadata | null>(null);
  const [query, setQuery] = useState("");
  const [group, setGroup] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<Page | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    request<EvidenceMetadata>("/evidence")
      .then((v) => {
        if (active) setMeta(v);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    let active = true;
    setPage(null);
    setError("");
    const timer = setTimeout(() => {
      const params = new URLSearchParams({
        query,
        superclass: group,
        scope: sensory ? "sensory" : "all",
        offset: String(offset),
        limit: "30",
      });
      request<Page>(`/evidence/neurons?${params}`)
        .then((v) => {
          if (active) setPage(v);
        })
        .catch((e) => {
          if (active) setError(e.message);
        });
    }, 200);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [query, group, offset, sensory]);
  return (
    <section className="panel neuron-evidence">
      <div className="panel-heading">
        <div>
          <h2>{sensory ? "Sensory inputs" : "All neurons & wiring"}</h2>
          <p>
            {meta
              ? (sensory ? meta.sensory_total : meta.total).toLocaleString()
              : "…"}{" "}
            imported neurons · source records and explicit assumptions
          </p>
        </div>
      </div>
      <p className="body-copy">
        {sensory
          ? "Inspect how the simulated environment reaches each sensory neuron. Class membership alone does not establish a working sensor; routes show their profile, modality switch and unresolved assignments."
          : "MaleCNS supports the recorded anatomy. Transmitter predictions retain their source scores; simulated electrical behavior has separate assumptions. These are field-based explanations, not independent literature reviews of every neuron."}
      </p>
      {meta && !sensory && (
        <div className="evidence-stats">
          <span>
            <strong>{meta.manifest.edges.toLocaleString()}</strong> imported
            directed pairs
          </span>
          <span>
            <strong>{meta.missing_type.toLocaleString()}</strong> without a type
            label
          </span>
          <span>
            <strong>{meta.missing_transmitter_score.toLocaleString()}</strong>{" "}
            without a transmitter score
          </span>
        </div>
      )}
      <div className="lab-controls">
        <label>
          Search neurons
          <input
            aria-label="Search neuron evidence"
            placeholder="Body ID, type, class, transmitter or nerve"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOffset(0);
              setSelected(null);
            }}
          />
        </label>
        <label>
          Source superclass
          <Select
            aria-label="Evidence superclass"
            value={group}
            onChange={(v) => {
              setGroup(v);
              setOffset(0);
              setSelected(null);
            }}
            options={[
              { value: "", label: "All superclasses" },
              ...Object.entries(meta?.classes ?? {}).map(([value, count]) => ({
                value,
                label: `${value} (${count.toLocaleString()})`,
              })),
            ]}
          />
        </label>
      </div>
      {error && <p role="alert">{error}</p>}
      {!page && !error && <p role="status">Loading neuron records…</p>}
      {page && (
        <>
          <p className="body-copy" role="status">
            {page.total.toLocaleString()} matching neurons ·{" "}
            {page.total
              ? `${page.offset + 1}–${Math.min(page.offset + 30, page.total)}`
              : "0"}
          </p>
          <div className="evidence-list">
            {page.records.map((row) => (
              <div key={row.body_id} className="evidence-row">
                <button
                  aria-expanded={selected === row.body_id}
                  onClick={() =>
                    setSelected(selected === row.body_id ? null : row.body_id)
                  }
                >
                  <span>
                    <strong>{row.type || "Type unspecified"}</strong> ·{" "}
                    {row.body_id}
                    <small>
                      {row.superclass} ·{" "}
                      {row.transmitter || "Transmitter unresolved"}
                      {row.transmitter_score !== null
                        ? ` · prediction score ${row.transmitter_score}`
                        : " · no prediction score"}
                    </small>
                  </span>
                  <span>{selected === row.body_id ? "Close" : "Inspect"}</span>
                </button>
                {selected === row.body_id && (
                  <NeuronDetail bodyId={row.body_id} />
                )}
              </div>
            ))}
          </div>
          {!page.total && <p>No neurons match these filters.</p>}
          <div className="lab-controls evidence-pagination">
            <button
              disabled={offset === 0}
              onClick={() => {
                setOffset(Math.max(0, offset - 30));
                setSelected(null);
              }}
            >
              Previous page
            </button>
            <span>
              Page {Math.floor(page.offset / 30) + 1} of{" "}
              {Math.max(1, Math.ceil(page.total / 30))}
            </span>
            <button
              disabled={offset + 30 >= page.total}
              onClick={() => {
                setOffset(offset + 30);
                setSelected(null);
              }}
            >
              Next page
            </button>
            <button
              onClick={() =>
                downloadJSON("malecns-neuron-evidence-page.json", {
                  scope: sensory ? "sensory" : "all",
                  query,
                  superclass: group,
                  ...page,
                })
              }
            >
              Export this page
            </button>
          </div>
        </>
      )}
      {meta && (
        <details>
          <summary>Dataset boundary & source files</summary>
          <p className="body-copy">
            {meta.manifest.selection}{" "}
            {meta.manifest.excluded_annotation_rows.toLocaleString()}{" "}
            unclassified source rows and{" "}
            {meta.manifest.boundary_connection_rows.toLocaleString()} crossing
            connection rows are outside the imported graph.
          </p>
          <p className="body-copy">{meta.confidence_policy}</p>
          {meta.manifest.sources.map((s) => (
            <p className="body-copy evidence-source" key={s.file}>
              <a href={s.url} target="_blank" rel="noreferrer">
                {s.file}
              </a>
              <small>SHA256 {s.sha256}</small>
            </p>
          ))}
        </details>
      )}
    </section>
  );
}

function NeuronDetail({ bodyId }: { bodyId: number }) {
  const [data, setData] = useState<EvidenceDetail | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    request<EvidenceDetail>(`/evidence/neurons/${bodyId}`)
      .then((v) => {
        if (active) setData(v);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [bodyId]);
  if (error) return <p role="alert">{error}</p>;
  if (!data) return <p role="status">Loading evidence…</p>;
  return (
    <div className="evidence-detail">
      {data.claims.map((c) => (
        <div key={c.name} className="evidence-claim">
          <h3>
            {c.name} <span>{c.level}</span>
          </h3>
          <p className="body-copy">{c.basis}</p>
        </div>
      ))}
      <h3>Body and environment interfaces</h3>
      <p className="body-copy">{data.sensory_explanation}</p>
      {data.sensory_inputs.map((s, i) => (
        <div className="evidence-claim" key={i}>
          <strong>
            {s.mode} · {s.confidence}
          </strong>
          <p className="body-copy">
            {s.basis} Profile {s.selected ? "selected" : "not selected"};
            modality{" "}
            {s.enabled === null
              ? "depends on experiment"
              : s.enabled
                ? "enabled"
                : "disabled"}
            .
          </p>
          {s.routing && <pre>{JSON.stringify(s.routing, null, 2)}</pre>}
          {s.source && (
            <a href={s.source} target="_blank" rel="noreferrer">
              Input reference
            </a>
          )}
        </div>
      ))}
      {data.motor_output ? (
        <MotorEvidence records={[data.motor_output]} />
      ) : (
        <p className="body-copy">
          No motor-to-muscle output is assigned to this neuron.
        </p>
      )}
      <p className="body-copy">{data.intervention}</p>
      <WiringEvidence bodyId={bodyId} />
      <details>
        <summary>All retained annotation fields</summary>
        <dl className="annotation-fields">
          {Object.entries(data.annotations).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>
                {value == null
                  ? "Not provided"
                  : typeof value === "object"
                    ? JSON.stringify(value)
                    : String(value)}
              </dd>
            </div>
          ))}
        </dl>
      </details>
      <details>
        <summary>Configured physiology assumptions</summary>
        <pre>{JSON.stringify(data.physiology, null, 2)}</pre>
      </details>
      <details>
        <summary>Sources & checksums</summary>
        {data.sources.map((s) => (
          <p key={s.url} className="body-copy evidence-source">
            <a href={s.url} target="_blank" rel="noreferrer">
              {s.title}
            </a>
            {s.sha256 && <small>SHA256 {s.sha256}</small>}
          </p>
        ))}
      </details>
      <button
        onClick={() =>
          downloadJSON(`malecns-neuron-${bodyId}-evidence.json`, data)
        }
      >
        Export neuron evidence
      </button>
    </div>
  );
}

function WiringEvidence({ bodyId }: { bodyId: number }) {
  const [open, setOpen] = useState(false);
  const [direction, setDirection] = useState("incoming");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<Wiring | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!open) return;
    let active = true;
    setData(null);
    setError("");
    request<Wiring>(
      `/neural-view/neuron/${bodyId}?direction=${direction}&offset=${offset}&limit=30`,
    )
      .then((v) => {
        if (active) setData(v);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [open, bodyId, direction, offset]);
  return (
    <details onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Inspect measured neuron-to-neuron wiring</summary>
      <p className="body-copy">
        Imported anatomical pairs and integer synapse counts. No per-pair
        confidence score is available in this aggregate table. Counts do not
        measure electrical efficacy.
      </p>
      <div className="lab-controls">
        <Select
          aria-label="Evidence wiring direction"
          value={direction}
          onChange={(v) => {
            setDirection(v);
            setOffset(0);
          }}
          options={[
            { value: "incoming", label: "Incoming connections" },
            { value: "outgoing", label: "Outgoing connections" },
          ]}
        />
      </div>
      {error && <p role="alert">{error}</p>}
      {!data && !error && <p role="status">Loading connections…</p>}
      {data && (
        <>
          <p className="body-copy">
            {data.incoming_total.toLocaleString()} incoming partners /{" "}
            {data.incoming_synapses.toLocaleString()} synapses ·{" "}
            {data.outgoing_total.toLocaleString()} outgoing partners /{" "}
            {data.outgoing_synapses.toLocaleString()} synapses
          </p>
          {data.partners.map((p) => (
            <p className="body-copy" key={p.body_id}>
              {direction === "incoming"
                ? `${p.body_id} → ${bodyId}`
                : `${bodyId} → ${p.body_id}`}{" "}
              · {p.type || "Type unspecified"} · {p.synapses} synapses
            </p>
          ))}
          <div className="lab-controls">
            <button
              disabled={!offset}
              onClick={() => setOffset(Math.max(0, offset - 30))}
            >
              Previous connections
            </button>
            <span>
              {data.total ? offset + 1 : 0}–{Math.min(offset + 30, data.total)}{" "}
              / {data.total}
            </span>
            <button
              disabled={offset + 30 >= data.total}
              onClick={() => setOffset(offset + 30)}
            >
              Next connections
            </button>
          </div>
        </>
      )}
    </details>
  );
}

export function MotorEvidencePage() {
  const [records, setRecords] = useState<MotorMappingRecord[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    request<{
      available: boolean;
      mapping: MotorMappingRecord[];
      unmapped: MotorMappingRecord[];
    }>("/connectome/mapping")
      .then((v) => {
        if (active) {
          if (v.available) setRecords([...v.mapping, ...v.unmapped]);
          else setError("The measured motor mapping is not loaded.");
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  return (
    <section className="panel">
      <h2>Motor outputs</h2>
      <p className="body-copy">
        Each motor neuron retains its target evidence, original reference scores
        and mechanical assumptions. Unresolved outputs remain documented.
      </p>
      {error && <p role="alert">{error}</p>}
      {records ? (
        <MotorEvidence records={records} />
      ) : (
        !error && <p role="status">Loading motor evidence…</p>
      )}
    </section>
  );
}
