import { Component, lazy, Suspense, useCallback, useState } from "react";
import type { ReactNode, ErrorInfo } from "react";
import {
  Bug,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  X,
  WifiOff,
} from "lucide-react";
import { useWorkbench, downloadJSON } from "./lib/api";
import { Telemetry } from "./components/Telemetry";
const FlyScene = lazy(() =>
  import("./components/Scene").then((x) => ({ default: x.FlyScene })),
);
const Brain = lazy(() =>
  import("./components/Brain").then((x) => ({ default: x.Brain })),
);
const Training = lazy(() =>
  import("./components/Training").then((x) => ({ default: x.Training })),
);
const DataPanel = lazy(() =>
  import("./components/DataPanel").then((x) => ({ default: x.DataPanel })),
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
          {w.connected ? "PyTorch connected" : "Connecting to PyTorch"}
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
              <strong>Odor conditioning</strong>
            </div>
            <div className="transport">
              <button
                className="primary"
                disabled={!w.connected}
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
                disabled={!w.connected}
                onClick={() => {
                  w.setReplay(null);
                  void w.command("step");
                }}
              >
                <SkipForward size={16} /> Step
              </button>
              <button
                disabled={!w.connected}
                onClick={() => w.command("reset")}
              >
                <RotateCcw size={16} /> Reset
              </button>
            </div>
            <label className="odor-control">
              Sensory cue
              <select
                aria-label="Sensory cue"
                value={s?.odor ?? "A"}
                disabled={!w.connected}
                onChange={(e) =>
                  w.command("odor", {
                    odor: e.target.value,
                    intensity: s?.intensity ?? 0.7,
                  })
                }
              >
                <option value="A">Odor A</option>
                <option value="B">Odor B</option>
                <option value="none">No odor</option>
              </select>
            </label>
            <label className="speed-control">
              Rate
              <select
                aria-label="Simulation rate"
                value={s?.speed ?? 1}
                disabled={!w.connected}
                onChange={(e) => w.command("speed", { speed: +e.target.value })}
              >
                <option value="1">1×</option>
                <option value="2">2×</option>
                <option value="4">4×</option>
              </select>
            </label>
          </div>
          <main className="experiment-main">
            <div className="model-notice">
              <span>
                <i /> Synthetic reference circuit
              </span>
              <span>
                Measured MaleCNS wiring is available in{" "}
                <button onClick={() => setTab("Data & model")}>
                  Data & model →
                </button>
              </span>
            </div>
            {d && s ? (
              <>
                <div className="simulation-grid">
                  <SceneBoundary>
                    <Suspense
                      fallback={
                        <div className="panel loading">Preparing 3D body…</div>
                      }
                    >
                      <FlyScene simulation={d} />
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
                        disabled={!w.connected || w.replay !== null}
                      />
                    </Suspense>
                  </SceneBoundary>
                </div>
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
            <span className="model-tag">Synthetic reference model</span>
          </div>
          <Suspense
            fallback={<div className="panel loading">Loading training…</div>}
          >
            <Training
              training={w.training}
              command={w.command}
              disabled={!w.connected}
              onError={onError}
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
          Neural steps <strong>{(d?.steps ?? 0).toLocaleString()}</strong>
        </span>
        <span>
          Contact points <strong>{d?.body.contacts ?? 0}</strong>
        </span>
        <span className="footer-model">
          {d?.model.neurons ?? 96} modeled units · CPU
        </span>
        <span className="status-word">
          <i className={s?.running ? "running" : ""} />
          {w.replay !== null ? "Replay" : s?.running ? "Running" : "Paused"}
        </span>
      </footer>
    </div>
  );
}
