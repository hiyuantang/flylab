import { useEffect, useState } from "react";
import { Database, ExternalLink, Zap, Check, Download } from "lucide-react";
import type { Dataset, Probe } from "../lib/types";
import { request, downloadJSON } from "../lib/api";
import { LineChart } from "./Telemetry";
export function DataPanel({ onError }: { onError: (s: string) => void }) {
  const [data, setData] = useState<Dataset | null>(null);
  const [neuron, setNeuron] = useState(10001);
  const [amplitude, setAmplitude] = useState(2);
  const [probe, setProbe] = useState<Probe | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    request<Dataset>("/data")
      .then(setData)
      .catch((e) => onError(e.message));
  }, [onError]);
  const run = async () => {
    setBusy(true);
    try {
      setProbe(
        await request<Probe>("/probe", {
          body_id: neuron,
          amplitude,
          duration_ms: 100,
        }),
      );
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="data-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Model provenance</h2>
            <p>Know what is measured, and what is modeled.</p>
          </div>
          <Database size={21} />
        </div>
        <div className="model-comparison">
          <div>
            <span className="model-label">EMBODIED EXPERIMENT</span>
            <h3>Synthetic reference circuit</h3>
            <p>
              96 rate units drive a connected NeuroMechFly body with 69 segments
              and 84 effective muscle actuators. Anatomy comes from a female
              specimen; neural wiring, muscle mappings and walking control are
              modeled assumptions. Feet provide contact feedback.
            </p>
            <span className="status-line">
              <Check size={14} /> PyTorch + MuJoCo · working closed loop
            </span>
          </div>
          <div>
            <span className="model-label">MEASURED CONNECTIVITY PROBE</span>
            <h3>MaleCNS v1.0</h3>
            <p>
              Actual neuron IDs and synapse counts from Janelia. The isolated
              subgraph uses assumed leaky integrate-and-fire dynamics. It is not
              connected to the embodied model's muscles.
            </p>
            <span className="status-line">
              <Check size={14} /> Measured wiring · assumed physiology
            </span>
          </div>
        </div>
        <div className="scientific-note">
          <strong>Next scientific integration</strong>
          <p>
            Matching identified motor neurons to specific muscles, calibrating
            neuron and receptor dynamics, and validating responses against
            experiments are required before the measured circuit can drive the
            body with biological claims.
          </p>
        </div>
      </section>
      {data?.neurons ? (
        <>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Imported MaleCNS subgraph</h2>
                <p>
                  Seed {data.seed} ·{" "}
                  {data.neurons.find((n) => n.body_id === data.seed)?.type}
                </p>
              </div>
              <a
                className="button-link"
                href="https://male-cns.janelia.org/download/"
                target="_blank"
                rel="noreferrer"
              >
                Source <ExternalLink size={14} />
              </a>
            </div>
            <div className="dataset-metrics">
              <div>
                <strong>{data.neurons.length.toLocaleString()}</strong>
                <span>Selected neurons</span>
              </div>
              <div>
                <strong>{data.edge_count.toLocaleString()}</strong>
                <span>Directed connections</span>
              </div>
              <div>
                <strong>{data.synapse_count.toLocaleString()}</strong>
                <span>Synapse count</span>
              </div>
              <div>
                <strong>
                  {data.boundary_connection_rows.toLocaleString()}
                </strong>
                <span>Excluded boundary edges</span>
              </div>
            </div>
            <p className="body-copy">{data.selection}</p>
            <details>
              <summary>Source files & checksums</summary>
              <p className="body-copy">
                {data.annotation_rows.toLocaleString()} annotation rows ·{" "}
                {data.source_connection_rows.toLocaleString()} segment-level
                connection rows. Segment rows are not equivalent to identified
                neurons. License: CC-BY.
              </p>
              {data.sources.map((s) => (
                <div className="source-file" key={s.file}>
                  <a href={s.url}>
                    {s.file} <ExternalLink size={12} />
                  </a>
                  <span>{(s.bytes / 1e6).toFixed(1)} MB</span>
                  <code>SHA256 {s.sha256}</code>
                </div>
              ))}
            </details>
          </section>
          <section className="panel probe-panel">
            <div className="panel-heading">
              <div>
                <h2>Stimulate measured wiring</h2>
                <p>500 ms simulation · 1 ms integration · 100 ms stimulus</p>
              </div>
              <Zap size={21} />
            </div>
            <div className="probe-controls">
              <label>
                Neuron
                <select
                  aria-label="Probe neuron"
                  value={neuron}
                  onChange={(e) => setNeuron(+e.target.value)}
                >
                  {data.neurons.map((n) => (
                    <option key={n.body_id} value={n.body_id}>
                      {n.type ?? "Untyped"} · {n.body_id}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Current amplitude
                <input
                  aria-label="Probe amplitude"
                  type="number"
                  min="0"
                  max="5"
                  step=".5"
                  value={amplitude}
                  onChange={(e) => setAmplitude(+e.target.value)}
                />
              </label>
              <button
                className="primary"
                disabled={busy || amplitude < 0 || amplitude > 5}
                onClick={run}
              >
                <Zap size={16} />
                {busy ? "Simulating…" : "Run neural probe"}
              </button>
            </div>
            {probe ? (
              <>
                <div className="probe-result">
                  <div>
                    <strong>{probe.active_neurons}</strong>
                    <span>Neurons that spiked</span>
                  </div>
                  <div>
                    <strong>{probe.spikes}</strong>
                    <span>Total spikes</span>
                  </div>
                  <button
                    onClick={() => downloadJSON("malecns-probe.json", probe)}
                  >
                    <Download size={15} /> Export probe
                  </button>
                </div>
                <div className="chart-legend">
                  <span>
                    <i style={{ background: "#65aeff" }} /> Stimulated neuron
                    voltage (relative threshold)
                  </span>
                </div>
                <LineChart series={[probe.history.map((x) => x.voltage)]} />
                <div className="axis">
                  <span>0 ms</span>
                  <span>500 ms</span>
                </div>
                <div className="probe-neurons">
                  {probe.neurons.slice(0, 12).map((n) => (
                    <div key={n.body_id}>
                      <span>
                        {n.type ?? "Untyped"} <small>{n.body_id}</small>
                      </span>
                      <span className="mono">{n.spikes} spikes</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="probe-empty">
                Select an identified neuron to test signal propagation through
                its measured connections.
              </div>
            )}
            <details>
              <summary>Firing-model assumptions</summary>
              <p className="body-copy">{data.assumptions}</p>
            </details>
          </section>
        </>
      ) : (
        <section className="panel body-copy">
          {data
            ? "Measured subgraph is not prepared. Run the importer documented in README."
            : "Loading dataset manifest…"}
        </section>
      )}
    </div>
  );
}
