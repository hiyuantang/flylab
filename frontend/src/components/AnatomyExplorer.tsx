import { useEffect, useState } from "react";
import { ArrowLeft, ChevronRight, RefreshCw, Search } from "lucide-react";
import type { AnatomySnapshot } from "../lib/types";

const number = (value: number) => value.toLocaleString();
const hz = (value: number) => value.toFixed(2);
type View = AnatomySnapshot["view"];

export function AnatomyExplorer({ available }: { available: boolean }) {
  const [selection, setSelection] = useState({
    group: "all",
    view: "groups" as View,
    query: "",
    offset: 0,
  });
  const [data, setData] = useState<AnatomySnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    if (!available) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    // Sequential low-rate polling samples every neuron; only display rows paginate.
    const load = async () => {
      try {
        const params = new URLSearchParams({
          ...selection,
          offset: String(selection.offset),
          limit: "12",
        });
        const response = await fetch(`/api/anatomy?${params}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          const message = await response.json();
          throw new Error(
            typeof message.detail === "string"
              ? message.detail
              : "Unable to inspect this group.",
          );
        }
        const result: AnatomySnapshot = await response.json();
        if (!controller.signal.aborted) {
          setData(result);
          setError(null);
        }
      } catch (e) {
        if (!controller.signal.aborted) setError((e as Error).message);
      } finally {
        if (!controller.signal.aborted) {
          setBusy(false);
          timer = setTimeout(load, 2000);
        }
      }
    };
    timer = setTimeout(load, selection.query ? 200 : 0);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [available, selection, refresh]);
  const navigate = (group: string, view: View = "groups") => {
    setBusy(true);
    setData(null);
    setError(null);
    setSelection({ group, view, query: "", offset: 0 });
  };
  const changeView = (view: View) => navigate(selection.group, view);
  const current = data?.group;
  return (
    <div className="anatomy-explorer" aria-label="Anatomical group explorer">
      <div className="anatomy-intro">
        <strong>Explore the measured network</strong>
        <p>
          Superclass → cell type → soma side → assigned column. Each neuron
          keeps its own state and connections.
        </p>
      </div>
      {!available ? (
        <p role="status">
          Connect to the live measured brain to explore its groups.
        </p>
      ) : (
        <>
          <nav className="anatomy-trail" aria-label="Anatomy hierarchy">
            {data?.breadcrumbs.map((group, i) => (
              <span key={group.id}>
                {i > 0 && <ChevronRight size={11} />}
                <button
                  onClick={() => navigate(group.id)}
                  disabled={group.id === selection.group}
                  title={group.label}
                >
                  {group.id === "all" ? "All neurons" : group.label}
                </button>
              </span>
            ))}
          </nav>
          <div className="anatomy-heading">
            <div>
              <small>{current?.kind_label ?? "Anatomical groups"}</small>
              <h3>{current?.label ?? "Loading group…"}</h3>
            </div>
            <button
              aria-label="Refresh anatomical activity"
              title="Refresh activity"
              onClick={() => {
                setBusy(true);
                setRefresh((x) => x + 1);
              }}
              disabled={busy}
            >
              <RefreshCw size={14} />
            </button>
          </div>
          {current && (
            <div className="anatomy-metrics">
              <span>
                <strong>{number(current.count)}</strong>neurons
              </span>
              <span>
                <strong>
                  {hz(current.mean_hz)} <small>Hz</small>
                </strong>
                mean rate
              </span>
              <span>
                <strong>{number(current.active_neurons)}</strong>above 1 Hz
              </span>
            </div>
          )}
          <div
            className="anatomy-tabs"
            role="group"
            aria-label="Group detail view"
          >
            {(["groups", "neurons", "wiring"] as const).map((view) => (
              <button
                key={view}
                aria-pressed={selection.view === view}
                onClick={() => changeView(view)}
              >
                {view === "groups"
                  ? "Subgroups"
                  : view === "neurons"
                    ? "Neurons"
                    : "Connections"}
              </button>
            ))}
          </div>
          {selection.view !== "wiring" && (
            <label className="anatomy-search">
              <Search size={14} />
              <input
                aria-label="Search anatomical groups or neurons"
                placeholder={
                  selection.view === "groups"
                    ? "Find a subgroup…"
                    : "Find a body ID or cell type…"
                }
                value={selection.query}
                onChange={(e) => {
                  setBusy(true);
                  setData(null);
                  setSelection((s) => ({
                    ...s,
                    query: e.target.value,
                    offset: 0,
                  }));
                }}
              />
            </label>
          )}
          {error && (
            <p role="alert" className="anatomy-error">
              {error}
            </p>
          )}
          {busy && !data && (
            <p role="status" className="anatomy-empty">
              Reading the measured network…
            </p>
          )}
          {data && (
            <>
              {selection.view === "groups" && (
                <div className="anatomy-group-list">
                  {data.items.map(
                    (item) =>
                      "count" in item && (
                        <button
                          key={item.id}
                          className="anatomy-group-row"
                          onClick={() => navigate(item.id)}
                        >
                          <span className="anatomy-group-label">
                            <strong>{item.label}</strong>
                            <small>{number(item.count)} neurons</small>
                          </span>
                          <span className="anatomy-rate">
                            <span className="bar-track">
                              <span
                                style={{
                                  width: `${Math.min(100, item.mean_hz)}%`,
                                }}
                              />
                            </span>
                            <small>{hz(item.mean_hz)} Hz</small>
                          </span>
                          <ChevronRight size={14} />
                        </button>
                      ),
                  )}
                  {data.total === 0 && (
                    <div className="anatomy-empty">
                      {selection.query ? (
                        "No matching subgroups."
                      ) : (
                        <>
                          This is the last annotation level.{" "}
                          <button onClick={() => changeView("neurons")}>
                            Inspect individual neurons
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
              {selection.view === "neurons" && (
                <div className="anatomy-neuron-list">
                  {data.items.map(
                    (item) =>
                      "body_id" in item && (
                        <button
                          className="anatomy-neuron-row"
                          key={item.body_id}
                          onClick={() => navigate(item.id, "wiring")}
                          title="Inspect this neuron's connections"
                        >
                          <span>
                            <strong>{item.type ?? "Unassigned type"}</strong>
                            <small>
                              Body {item.body_id} · soma{" "}
                              {item.soma_side ?? "unassigned"}
                            </small>
                            <small>
                              {item.transmitter ?? "Unknown transmitter"}
                              {item.hex.every((x) => x != null)
                                ? ` · hex ${item.hex.join(", ")}`
                                : ""}
                            </small>
                          </span>
                          <span className="anatomy-neuron-rate">
                            <strong>{hz(item.rate_hz)} Hz</strong>
                            <small>
                              {hz(item.voltage)} {item.voltage_unit}
                            </small>
                            <ChevronRight size={12} />
                          </span>
                        </button>
                      ),
                  )}
                  {data.total === 0 && (
                    <p className="anatomy-empty">No matching neurons.</p>
                  )}
                </div>
              )}
              {data.wiring && (
                <div className="anatomy-wiring">
                  <p>
                    Exact directed pairs and synapse counts. Incoming and
                    outgoing totals cross the selected group’s boundary.
                  </p>
                  <div className="anatomy-edge-totals">
                    {(["internal", "incoming", "outgoing"] as const).map(
                      (key) => (
                        <span key={key}>
                          <small>{key}</small>
                          <strong>
                            {number(data.wiring![key].connections)}
                          </strong>
                          <small>
                            {number(data.wiring![key].synapses)} synapses
                          </small>
                        </span>
                      ),
                    )}
                  </div>
                  <div className="anatomy-route-scroll">
                    <table>
                      <caption>Other endpoint’s superclass · synapses</caption>
                      <thead>
                        <tr>
                          <th>Superclass</th>
                          <th>Into selection</th>
                          <th>Out of selection</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.wiring.routes.map((row) => (
                          <tr key={row.label}>
                            <th>{row.label}</th>
                            <td>{number(row.incoming_synapses)}</td>
                            <td>{number(row.outgoing_synapses)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {data.wiring.routes.length === 0 && (
                    <p className="anatomy-empty">
                      No connections cross this selection’s boundary within the
                      imported graph.
                    </p>
                  )}
                </div>
              )}
              {selection.view !== "wiring" && (
                <div className="anatomy-pagination">
                  <button
                    aria-label="Previous anatomical results"
                    disabled={selection.offset === 0}
                    onClick={() => {
                      setBusy(true);
                      setData(null);
                      setSelection((s) => ({
                        ...s,
                        offset: Math.max(0, s.offset - data.limit),
                      }));
                    }}
                  >
                    Previous
                  </button>
                  <span>
                    {data.total
                      ? `${data.offset + 1}–${Math.min(data.offset + data.limit, data.total)} of ${number(data.total)}`
                      : "0 results"}
                  </span>
                  <button
                    aria-label="Next anatomical results"
                    disabled={selection.offset + data.limit >= data.total}
                    onClick={() => {
                      setBusy(true);
                      setData(null);
                      setSelection((s) => ({
                        ...s,
                        offset: s.offset + data.limit,
                      }));
                    }}
                  >
                    Next
                  </button>
                </div>
              )}
              <div className="anatomy-footnote">
                <span>
                  Activity at {(data.neural_time_ms / 1000).toFixed(2)} s
                  simulated · refreshes every 2 s
                </span>
                <span>
                  Group bars: 0–100 Hz; values above 100 Hz remain in the
                  numbers.
                </span>
              </div>
              <details className="anatomy-provenance">
                <summary>Membership & source</summary>
                <p>{data.membership}</p>
                <p>{data.limitations}</p>
                <p>{data.source}</p>
                <p className="mono">Graph: {data.graph_sha256}</p>
                <p className="mono">Annotations: {data.annotation_sha256}</p>
              </details>
            </>
          )}
          {current?.parent && (
            <button
              className="anatomy-back"
              onClick={() => navigate(current.parent!)}
            >
              <ArrowLeft size={12} /> Back to parent
            </button>
          )}
        </>
      )}
    </div>
  );
}
