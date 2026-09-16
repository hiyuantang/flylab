import { useViewState, oneOf } from "../lib/viewState";
import { useEffect, useState } from "react";
import { Database, Download, Play, Square } from "lucide-react";
import { request, downloadJSON } from "../lib/api";
import { NeuronEvidence, MotorEvidencePage } from "./NeuronEvidence";
import { ModelControls } from "./ModelControls";

type PaperResult = {
  dataset: string;
  neurons: number;
  connection_rows: number;
  duration_ms: number;
  seed: number;
  cancelled?: boolean;
  limits: string;
  results: {
    condition: string;
    mn9_hz: number[];
    total_spikes: number;
    wall_seconds: number;
  }[];
};
type PaperStatus = {
  running: boolean;
  error: string | null;
  progress: {
    condition: string;
    simulated_ms: number;
    target_ms: number;
    neurons: number;
  } | null;
  result: PaperResult | null;
};
export function DataPanel({ onError }: { onError: (s: string) => void }) {
  const [page, setPage] = useViewState(
    "data.page",
    "overview",
    oneOf(["overview", "neurons", "sensory", "motor", "reference", "model"]),
  );
  const [paper, setPaper] = useState<PaperStatus | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (page !== "reference") return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const value = await request<PaperStatus>("/paper");
        if (!disposed) setPaper(value);
      } catch (e) {
        if (!disposed) onError((e as Error).message);
      }
      if (!disposed) timer = setTimeout(poll, 1500);
    };
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [onError, page]);
  const run = async () => {
    setBusy(true);
    try {
      setPaper(
        await request<PaperStatus>(
          paper?.running ? "/paper/stop" : "/paper/start",
          {},
        ),
      );
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const result = paper?.result;
  return (
    <div className="data-layout">
      <nav className="evidence-nav" aria-label="Data and model pages">
        {[
          ["overview", "Overview"],
          ["neurons", "Neurons & wiring"],
          ["sensory", "Sensory inputs"],
          ["motor", "Motor outputs"],
          ["model", "Model settings"],
          ["reference", "Paper reference"],
        ].map(([value, label]) => (
          <button
            key={value}
            aria-current={page === value ? "page" : undefined}
            onClick={() => setPage(value)}
          >
            {label}
          </button>
        ))}
      </nav>
      {page === "neurons" && <NeuronEvidence key="all" />}
      {page === "sensory" && <NeuronEvidence key="sensory" sensory />}
      {page === "motor" && <MotorEvidencePage />}
      {page === "overview" && (
        <>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>What the model preserves</h2>
                <p>
                  Full imported populations, explicit biological assumptions.
                </p>
              </div>
              <Database size={21} />
            </div>
            <div className="model-comparison">
              <div>
                <span className="model-label">EMBODIED MALECNS</span>
                <h3>Measured wiring, continuous neural state</h3>
                <p>
                  Every imported neuron updates on the selected neural clock.
                  Raw synapse counts set efficacy, with explicit delays and
                  refractory periods. Performance never changes population size
                  or time step. Precision changes only when you explicitly
                  switch execution mode. The schematic brain view summarizes
                  activity; it does not reduce the simulated network.
                </p>
              </div>
              <div>
                <span className="model-label">MODEL LIMITS</span>
                <h3>Anatomy is not complete physiology</h3>
                <p>
                  Unknown transmitter effects, retinal routing, and many muscle
                  targets remain unresolved. The body uses female-derived
                  anatomy and assumed muscle properties. Unclassified source
                  segments are disclosed on the Neurons & wiring page. This is a
                  scientific model, not a validated digital organism.
                </p>
              </div>
            </div>
            <p className="body-copy">
              The CPU reference uses float64. Experimental MPS GPU execution
              uses float32 or explicitly selected float16 and can change spike
              timing. Slow hardware produces slow motion; neural and physical
              time stay synchronized. Save state preserves the full brain,
              pending spikes, random generator, and body for continuation.
            </p>
          </section>
        </>
      )}
      {page === "reference" && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Paper reference experiment</h2>
              <p>Shiu et al., Nature 2024 · full FlyWire female v630 graph</p>
            </div>
            <a
              className="button-link"
              href="https://www.nature.com/articles/s41586-024-07763-9"
              target="_blank"
              rel="noreferrer"
            >
              Read paper
            </a>
          </div>
          <p className="body-copy">
            Run no-input, sugar, and sugar + bitter conditions on all 127,400
            source neurons, without training. Measure both MN9 feeding outputs.
            This isolated reference uses the paper’s specimen and signed counts;
            it is not the MaleCNS animal in Experiment.
          </p>
          <div className="lab-controls">
            <button className="primary" disabled={busy || !paper} onClick={run}>
              {paper?.running ? <Square size={15} /> : <Play size={15} />}
              {paper?.running ? "Stop experiment" : "Run full-graph reference"}
            </button>
            {result && (
              <button
                onClick={() =>
                  downloadJSON("flylab-paper-reference.json", result)
                }
              >
                <Download size={15} /> Export results
              </button>
            )}
            <span className="body-copy">
              1 s per condition · seed 42 · one trial
            </span>
          </div>
          {paper?.running && (
            <p role="status" className="body-copy">
              {paper.progress
                ? `${paper.progress.condition}: ${paper.progress.simulated_ms} / ${paper.progress.target_ms} ms · ${paper.progress.neurons.toLocaleString()} neurons`
                : "Loading complete source graph…"}
            </p>
          )}
          {paper?.error && (
            <p role="alert" className="body-copy">
              {paper.error}
            </p>
          )}
          {result && (
            <>
              {paper?.running && (
                <p className="body-copy">Previous completed result:</p>
              )}
              <div className="lab-table-scroll reference-results">
                <table aria-label="Paper reference results">
                  <thead>
                    <tr>
                      <th>Condition</th>
                      <th>MN9 output 1 (Hz)</th>
                      <th>MN9 output 2 (Hz)</th>
                      <th>Spikes</th>
                      <th>Compute time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.results.map((r) => (
                      <tr key={r.condition}>
                        <td>{r.condition}</td>
                        <td>{r.mn9_hz[0]}</td>
                        <td>{r.mn9_hz[1]}</td>
                        <td>{r.total_spikes.toLocaleString()}</td>
                        <td>{r.wall_seconds.toFixed(1)} s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="body-copy">
                {result.cancelled
                  ? "Experiment stopped. Partial results only."
                  : result.limits}
              </p>
            </>
          )}
        </section>
      )}
      {page === "model" && <ModelControls onError={onError} />}
    </div>
  );
}
