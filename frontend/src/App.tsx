import Select from "./components/Select";
import { Component, lazy, Suspense, useCallback, useState } from "react";
import type { ReactNode, ErrorInfo } from "react";
import {
  Bug,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Save,
  FolderOpen,
  X,
  WifiOff,
} from "lucide-react";
import { useWorkbench, downloadJSON } from "./lib/api";
import { Telemetry } from "./components/Telemetry";
import { SceneChooser } from "./components/World";
import { Senses } from "./components/Senses";
const FlyScene = lazy(() =>
  import("./components/Scene").then((x) => ({ default: x.FlyScene })),
);
const Brain = lazy(() =>
  import("./components/Brain").then((x) => ({ default: x.Brain })),
);
const DataPanel = lazy(() =>
  import("./components/DataPanel").then((x) => ({ default: x.DataPanel })),
);
const PhysicalLab = lazy(() =>
  import("./components/PhysicalLab").then((x) => ({ default: x.PhysicalLab })),
);
class SceneBoundary extends Component<
  { children: ReactNode },
  { error: boolean }
> {
  state = { error: false };
  static getDerivedStateFromError() {
    return { error: true };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack);
  }
  render() {
    return this.state.error ? (
      <div className="panel loading">
        <h2>3D view unavailable</h2>
        <p>
          Enable hardware acceleration and reload. The Python simulation remains
          available.
        </p>
      </div>
    ) : (
      this.props.children
    );
  }
}
export default function App() {
  const [tab, setTab] = useState("Experiment");
  const w = useWorkbench();
  const s = w.simulation;
  const d = w.display;
  const onError = useCallback(
    (message: string) => w.setError(message),
    [w.setError],
  );
  return (
    <div className="app-shell">
      <header className="app-header">
        <a href="/" className="brand" aria-label="FlyLab home">
          <Bug size={28} strokeWidth={1.3} />
          <strong>FlyLab</strong>
        </a>
        <span className="app-subtitle">Neural simulation workbench</span>
        <nav aria-label="Workspace">
          {["Experiment", "Training", "Data & model"].map((name) => (
            <button
              className={tab === name ? "active" : ""}
              key={name}
              onClick={() => setTab(name)}
            >
              {name}
            </button>
          ))}
        </nav>
        <div className={`connection ${w.connected ? "" : "offline"}`}>
          <i />
          {w.busy
            ? "Applying change…"
            : w.connected
              ? "PyTorch connected"
              : "Connecting to PyTorch"}
        </div>
      </header>
      {w.error && (
        <div className="error-banner" role="alert">
          {w.error}
          <button aria-label="Dismiss error" onClick={() => w.setError(null)}>
            <X size={16} />
          </button>
        </div>
      )}
      {w.notice && (
        <div className="save-notice" role="status">
          {w.notice}
        </div>
      )}
      {!w.connected && (
        <div className="offline-banner" role="status">
          <WifiOff size={15} />{" "}
          {s
            ? "Connection interrupted. Controls are paused until the backend reconnects."
            : "Connecting to the local simulation server…"}
        </div>
      )}
      {tab === "Experiment" && (
        <>
          <div className="experiment-toolbar">
            <div className="task-field">
              <span>Task</span>
              <strong>Full measured brain</strong>
            </div>
            <label className="environment-choice">
              Environment
              <Select
                aria-label="Environment"
                title="Changing environment resets the body"
                disabled={!w.connected || w.busy || !s?.model.ready}
                value={s?.environment?.mode ?? "uniform"}
                onChange={(value) => {
                  w.setReplay(null);
                  void w.command("environment", {
                    environment: value,
                  });
                }}
                options={[
                  { value: "uniform", label: "Uniform cue" },
                  { value: "spatial", label: "Spatial odor field" },
                ]}
              />
            </label>
            <div className="transport">
              <button
                className="primary"
                disabled={!w.connected || w.busy || !s?.model.ready}
                onClick={() => w.command(s?.running ? "pause" : "run")}
              >
                {s?.running ? (
                  <Pause size={16} fill="currentColor" />
                ) : (
                  <Play size={16} fill="currentColor" />
                )}
                {s?.running ? "Pause" : "Run"}
              </button>
              <button
                disabled={!w.connected || w.busy || !s?.model.ready}
                onClick={() => {
                  w.setReplay(null);
                  void w.command("step");
                }}
              >
                <SkipForward size={16} /> Step
              </button>
              <button
                disabled={!w.connected || w.busy || !s?.model.ready}
                onClick={() => w.command("reset")}
              >
                <RotateCcw size={16} /> Reset
              </button>
            </div>
            <label className="odor-control">
              Sensory cue
              <Select
                aria-label="Sensory cue"
                value={s?.odor ?? "A"}
                disabled={!w.connected || w.busy || !s?.model.ready}
                onChange={(value) =>
                  w.command("odor", {
                    odor: value,
                    intensity: s?.intensity ?? 0.7,
                  })
                }
                options={[
                  { value: "A", label: "Odor A" },
                  { value: "B", label: "Odor B" },
                  { value: "none", label: "No odor" },
                ]}
              />
            </label>
            <label className="speed-control">
              Pacing ceiling
              <Select
                aria-label="Simulation rate"
                value={s?.speed ?? 1}
                disabled={!w.connected || w.busy || !s?.model.ready}
                onChange={(value) => w.command("speed", { speed: +value })}
                options={[
                  { value: "1", label: "1× real time" },
                  { value: "2", label: "2× real time" },
                  { value: "4", label: "4× real time" },
                ]}
              />
            </label>
          </div>
          <main className="experiment-main">
            <div className="model-notice">
              <span>
                <i /> {d?.model.name ?? "Preparing controller"}
              </span>
              <span className="life-controls">
                <button
                  disabled={!w.connected || w.busy || !s?.model.ready}
                  onClick={() => w.command("live_save")}
                  title="Save every neuron's state, delayed spikes, and the physical body"
                >
                  <Save size={14} /> Save state
                </button>
                <button
                  disabled={!w.connected || w.busy}
                  onClick={() => {
                    w.setReplay(null);
                    void w.command("live_restore");
                  }}
                  title="Restore the last full state, paused"
                >
                  <FolderOpen size={14} /> Resume saved
                </button>
              </span>
            </div>
            {d && s ? (
              <>
                <div className="clock-strip" aria-label="Simulation timing">
                  <span>
                    Simulated <strong>{d.time.toFixed(2)} s</strong>
                  </span>
                  <span>
                    Computed in{" "}
                    <strong>
                      {(d.timing?.compute_wall_seconds ?? 0).toFixed(1)} s
                    </strong>
                  </span>
                  <span>
                    {d.timing?.simulated_per_wall_second != null
                      ? `${d.timing.simulated_per_wall_second.toFixed(3)}× real time`
                      : "Ready to simulate"}{" "}
                    · full graph
                  </span>
                </div>
                <SceneChooser
                  scene={d.scene}
                  command={w.command}
                  disabled={!w.connected || w.busy || w.replay !== null}
                />
                <div className="simulation-grid">
                  <SceneBoundary>
                    <Suspense
                      fallback={
                        <div className="panel loading">Preparing 3D body…</div>
                      }
                    >
                      <FlyScene
                        key={d.scene.id + d.scene.version}
                        simulation={d}
                      />
                    </Suspense>
                  </SceneBoundary>
                  <SceneBoundary>
                    <Suspense
                      fallback={
                        <div className="panel loading">
                          Preparing neural view…
                        </div>
                      }
                    >
                      <Brain
                        simulation={d}
                        command={w.command}
                        liveAvailable={
                          w.connected && w.replay === null && !!s.model.ready
                        }
                        disabled={
                          !w.connected ||
                          w.busy ||
                          w.replay !== null ||
                          d.model.controller === "posture"
                        }
                      />
                    </Suspense>
                  </SceneBoundary>
                </div>
                {d.senses && (
                  <Senses
                    key={d.episode}
                    frame={d.senses}
                    controller={d.model.controller ?? "connectome"}
                    command={w.command}
                    disabled={!w.connected || w.busy || w.replay !== null}
                  />
                )}
                <Telemetry
                  simulation={d}
                  history={s.history}
                  frames={w.frames.current.length}
                  replay={w.replay}
                  onReplay={w.setReplay}
                  onExport={() =>
                    downloadJSON("flylab-recording.json", {
                      model: s.model,
                      frames: w.frames.current,
                      signals: s.history,
                    })
                  }
                />
                <p className="body-copy">
                  Every imported neuron and connection remains in the
                  simulation. Fixed {d.timing?.neural_dt_ms ?? 0.1} ms neural
                  integration · {d.model.precision} ·{" "}
                  {d.model.spikes?.toLocaleString()} spikes. Measured wiring;
                  assumed physiology, sensory encoding, and partial muscle
                  routing. Walking is not validated.
                </p>
              </>
            ) : (
              <div className="loading panel">
                <Bug size={42} />
                <h2>Preparing your workbench</h2>
                <p>Waiting for the PyTorch and MuJoCo backend.</p>
              </div>
            )}
          </main>
        </>
      )}
      {tab === "Training" && (
        <main className="secondary-main">
          <div className="page-heading">
            <div>
              <h1>Learning laboratory</h1>
              <p>Train a brain. Keep the experiment reproducible.</p>
            </div>
            <span className="model-tag">Full-graph physical learning</span>
          </div>
          <Suspense
            fallback={<div className="panel loading">Loading training…</div>}
          >
            <PhysicalLab
              command={w.command}
              onError={onError}
              scene={s?.scene}
            />
          </Suspense>
        </main>
      )}
      {tab === "Data & model" && (
        <main className="secondary-main">
          <div className="page-heading">
            <div>
              <h1>From anatomy to a model</h1>
              <p>Measured data, explicit assumptions, traceable experiments.</p>
            </div>
          </div>
          <Suspense
            fallback={<div className="panel loading">Loading data…</div>}
          >
            <DataPanel onError={onError} />
          </Suspense>
        </main>
      )}
      <footer className="status-bar">
        <span>
          Simulation time <strong>{(d?.time ?? 0).toFixed(2)} s</strong>
        </span>
        <span>
          Compute time{" "}
          <strong>{(d?.timing?.compute_wall_seconds ?? 0).toFixed(1)} s</strong>
        </span>
        <span>
          Contact points <strong>{d?.body.contacts ?? 0}</strong>
        </span>
        <span className="footer-model">
          {(d?.model.neurons ?? 0).toLocaleString()} neurons ·{" "}
          {d?.model.device?.toUpperCase() ?? "—"} ·{" "}
          {d?.model.precision?.replace("torch.", "") ?? "—"}
        </span>
        <span className="status-word">
          <i className={s?.running ? "running" : ""} />
          {w.replay !== null ? "Replay" : s?.running ? "Running" : "Paused"}
          {d?.timing?.simulated_per_wall_second != null &&
            ` · ${d.timing.simulated_per_wall_second.toFixed(3)}× measured rate`}
        </span>
      </footer>
    </div>
  );
}
