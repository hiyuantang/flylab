import { useViewState, oneOf } from "./lib/viewState";
import Select from "./components/Select";
import { RewardControls } from "./components/RewardControls";
import { LocomotionControls } from "./components/LocomotionControls";
import { StandingControls } from "./components/StandingControls";
import { GestureLab } from "./components/GestureLab";
import {
  GestureControls,
  GestureModelControls,
} from "./components/GestureControls";
import { WeightVersions } from "./components/WeightVersions";
import { useGestures } from "./lib/gestures";
import {
  Component,
  lazy,
  Suspense,
  useCallback,
  useRef,
  useEffect,
} from "react";
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
  SlidersHorizontal,
  ChevronDown,
  Footprints,
  Eye,
  Activity,
  Sparkles,
} from "lucide-react";
import { useWorkbench, downloadJSON } from "./lib/api";
import { Telemetry } from "./components/Telemetry";
import { SCENE_CHOICES } from "./components/World";
import { Senses, VisionReadout } from "./components/Senses";
import { PerformanceMeter } from "./components/PerformanceMeter";
const FlyScene = lazy(() =>
  import("./components/Scene").then((x) => ({ default: x.FlyScene })),
);
const PolicyNetwork = lazy(() => import("./components/PolicyNetwork"));
const Brain = lazy(() =>
  import("./components/Brain").then((x) => ({ default: x.Brain })),
);
const DataPanel = lazy(() =>
  import("./components/DataPanel").then((x) => ({ default: x.DataPanel })),
);
const FlightLab = lazy(() => import("./components/FlightLab"));
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
// Measure the remaining desktop height; the scene keeps its original size.
function LiveOverview({ children }: { children: ReactNode }) {
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const fit = () => {
      if (!element.current) return;
      const top = element.current.getBoundingClientRect().top + window.scrollY;
      element.current.style.setProperty(
        "--overview-height",
        `${Math.max(520, window.innerHeight - top - 16)}px`,
      );
    };
    const observer = new ResizeObserver(fit);
    document
      .querySelectorAll(
        ".app-header, .experiment-toolbar, .error-banner, .offline-banner, .save-notice",
      )
      .forEach((node) => observer.observe(node));
    window.addEventListener("resize", fit);
    fit();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", fit);
    };
  }, []);
  return (
    <div className="live-overview" ref={element}>
      {children}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useViewState(
    "workspace",
    "Experiment",
    oneOf(["Experiment", "Training", "Flight lab", "Weights", "Data & model"]),
  );
  const settings = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const dismiss = (event: PointerEvent) => {
      if (
        event.target instanceof Node &&
        !settings.current?.contains(event.target) &&
        settings.current
      )
        settings.current.open = false;
    };
    window.addEventListener("pointerdown", dismiss);
    return () => window.removeEventListener("pointerdown", dismiss);
  }, []);
  const w = useWorkbench();
  const s = w.simulation;
  const d = w.display;
  const onError = useCallback(
    (message: string) => w.setError(message),
    [w.setError],
  );
  const gestureController = useGestures(onError);
  return (
    <div
      className={`app-shell organized-workbench ${tab === "Training" ? "training-workbench" : ""}`}
      id="workbench-top"
    >
      <header className="app-header">
        <a href="/" className="brand" aria-label="FlyLab home">
          <Bug size={28} strokeWidth={1.3} />
          <strong>FlyLab</strong>
        </a>
        <span className="app-subtitle">Robot training workbench</span>
        <nav aria-label="Workspace">
          {[
            "Experiment",
            "Training",
            "Flight lab",
            "Weights",
            "Data & model",
          ].map((name) => (
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
            <label className="scene-choice">
              Scene
              <Select
                aria-label="Scene"
                title="Changing scenes resets the experiment"
                value={d?.scene.id ?? "lab"}
                disabled={
                  !w.connected || w.busy || !s?.model.ready || w.replay !== null
                }
                onChange={(value) => w.command("scene", { scene_id: value })}
                options={SCENE_CHOICES.map(({ id, label }) => ({
                  value: id,
                  label,
                }))}
              />
            </label>
            <GestureModelControls
              controller={gestureController}
              loaded={s?.gesture_model}
            />
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
            <details
              className="quick-settings"
              ref={settings}
              onKeyDown={(event) => {
                if (
                  event.key === "Escape" &&
                  !event.defaultPrevented &&
                  settings.current
                ) {
                  settings.current.open = false;
                  settings.current.querySelector("summary")?.focus();
                }
              }}
            >
              <summary>
                <SlidersHorizontal size={15} /> Settings{" "}
                <ChevronDown size={14} />
              </summary>
              <div className="quick-settings-content">
                <h3>Experiment settings</h3>
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
                <p className="hint">
                  Changing the environment resets the body. Detailed sensory and
                  behavior controls are below the views.
                </p>
                <div className="life-controls">
                  <button
                    disabled={!w.connected || w.busy || !s?.model.ready}
                    onClick={() => w.command("live_save")}
                  >
                    <Save size={14} /> Save state
                  </button>
                  <button
                    disabled={!w.connected || w.busy}
                    onClick={() => {
                      w.setReplay(null);
                      void w.command("live_restore");
                    }}
                  >
                    <FolderOpen size={14} /> Resume saved
                  </button>
                </div>
              </div>
            </details>
          </div>
          <main className="experiment-main">
            {d && s ? (
              <>
                <LiveOverview>
                  <div className="simulation-grid">
                    <SceneBoundary>
                      <Suspense
                        fallback={
                          <div className="panel loading">
                            Preparing 3D body…
                          </div>
                        }
                      >
                        <FlyScene
                          persistenceKey="experiment"
                          key={d.scene.id + d.scene.version}
                          simulation={d}
                          gestureControls={
                            <GestureControls
                              selected={d.gesture_stimulus?.gesture}
                              disabled={
                                !w.connected ||
                                w.replay !== null ||
                                gestureController.busy
                              }
                              onSelect={(gesture) =>
                                gestureController.act("/gestures/hand", {
                                  gesture,
                                })
                              }
                            />
                          }
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
                        {d.model.controller === "transformer" ? (
                          <PolicyNetwork
                            activity={d.policy_activity}
                            parameters={d.model.parameters}
                            modelKey={d.gesture_model ?? "new"}
                          />
                        ) : (
                          <Brain
                            simulation={d}
                            command={w.command}
                            liveAvailable={
                              w.connected &&
                              w.replay === null &&
                              !!s.model.ready
                            }
                            disabled={
                              !w.connected ||
                              w.busy ||
                              w.replay !== null ||
                              d.model.controller === "posture"
                            }
                          />
                        )}
                      </Suspense>
                    </SceneBoundary>
                  </div>
                  <div className="observation-strip">
                    {d.senses && (
                      <VisionReadout
                        bodies={d.body.bodies}
                        frame={d.senses}
                        controller={d.model.controller ?? "connectome"}
                        command={w.command}
                        disabled={!w.connected || w.busy || w.replay !== null}
                      />
                    )}
                    <section
                      className="panel session-overview"
                      aria-label="Current simulation"
                    >
                      <div className="session-heading">
                        <h2>Simulation</h2>
                        <span className="model-tag">
                          {w.replay !== null
                            ? "Replay"
                            : s.running
                              ? "Running"
                              : "Paused"}
                        </span>
                      </div>
                      <dl
                        className="session-stats"
                        aria-label="Simulation timing and execution"
                      >
                        <div>
                          <dt>Simulated time</dt>
                          <dd>
                            {d.time.toFixed(2)} <small>s</small>
                          </dd>
                        </div>
                        <div>
                          <dt>Compute time</dt>
                          <dd>
                            {(d.timing?.compute_wall_seconds ?? 0).toFixed(1)}{" "}
                            <small>s</small>
                          </dd>
                        </div>
                        <div>
                          <dt>
                            {d.model.controller === "transformer"
                              ? "Parameters"
                              : "Neurons"}
                          </dt>
                          <dd>
                            {(
                              d.model.parameters ?? d.model.neurons
                            ).toLocaleString()}
                          </dd>
                        </div>
                        <div>
                          <dt>Execution</dt>
                          <dd>
                            {d.model.device?.toUpperCase()}{" "}
                            <small>
                              · {d.model.precision?.replace("torch.", "")}
                            </small>
                          </dd>
                        </div>
                      </dl>
                      <p className="session-rate">
                        {d.timing?.simulated_per_wall_second != null
                          ? `${d.timing.simulated_per_wall_second.toFixed(3)}× real time`
                          : "Ready to simulate"}{" "}
                        ·{" "}
                        {d.model.controller === "transformer"
                          ? "visual policy"
                          : "full graph"}
                      </p>
                      <PerformanceMeter simulation={s} />
                      <p className="hint">
                        {d.model.controller === "transformer"
                          ? "Learned visual policy · physical performance requires validation."
                          : "Measured wiring · experimental body coupling. Walking is not validated."}
                      </p>
                    </section>
                  </div>
                </LiveOverview>
                <section
                  className="experiment-tools"
                  aria-label="Experiment tools"
                >
                  <div className="tools-heading">
                    <h2>Experiment tools</h2>
                    <span>Open a section when you need it</span>
                    <a href="#workbench-top">Back to views ↑</a>
                  </div>
                  {d.model.controller !== "transformer" && (
                    <details className="tool-section">
                      <summary>
                        <Footprints size={18} />
                        <strong>Behavior trials</strong>
                        <span>Standing support & walking drive</span>
                        <ChevronDown size={16} />
                      </summary>
                      <div className="tool-content">
                        <StandingControls
                          simulation={d}
                          command={w.command}
                          disabled={!w.connected || w.busy || w.replay !== null}
                        />
                        <LocomotionControls
                          simulation={d}
                          command={w.command}
                          disabled={!w.connected || w.busy || w.replay !== null}
                        />
                      </div>
                    </details>
                  )}
                  <details className="tool-section">
                    <summary>
                      <Eye size={18} />
                      <strong>Sensory settings</strong>
                      <span>Vision, hearing, wind, touch & leg feedback</span>
                      <ChevronDown size={16} />
                    </summary>
                    <div className="tool-content">
                      {d.senses && (
                        <Senses
                          key={d.episode}
                          frame={d.senses}
                          controller={d.model.controller ?? "connectome"}
                          command={w.command}
                          disabled={!w.connected || w.busy || w.replay !== null}
                        />
                      )}
                    </div>
                  </details>
                  {d.model.controller !== "transformer" && (
                    <details className="tool-section">
                      <summary>
                        <Sparkles size={18} />
                        <strong>Dopamine & learning</strong>
                        <span>Reward, punishment & plasticity</span>
                        <ChevronDown size={16} />
                      </summary>
                      <div className="tool-content">
                        <RewardControls
                          simulation={d}
                          command={w.command}
                          disabled={!w.connected || w.busy || w.replay !== null}
                        />
                      </div>
                    </details>
                  )}
                  <details className="tool-section">
                    <summary>
                      <Activity size={18} />
                      <strong>Recording & diagnostics</strong>
                      <span>Signal traces, muscle activity & replay</span>
                      <ChevronDown size={16} />
                    </summary>
                    <div className="tool-content">
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
                        simulation. Fixed{" "}
                        {(d.timing?.neural_dt_ms ?? 0.1).toFixed(3)} ms neural
                        integration · {d.model.precision} ·{" "}
                        {d.model.spikes?.toLocaleString()} spikes. Measured
                        wiring; assumed physiology, sensory encoding, and
                        partial muscle routing. Walking is not validated.
                      </p>
                    </div>
                  </details>
                </section>
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
        <main className="training-main">
          <Suspense
            fallback={<div className="panel loading">Loading training…</div>}
          >
            <GestureLab
              controller={gestureController}
              simulation={d}
              legacyTools={
                <details className="training-legacy-tools">
                  <summary>Earlier physical gain search</summary>
                  <PhysicalLab
                    command={w.command}
                    onError={onError}
                    scene={s?.scene}
                  />
                </details>
              }
            />
          </Suspense>
        </main>
      )}
      {tab === "Flight lab" && (
        <Suspense
          fallback={<div className="panel loading">Loading flight lab…</div>}
        >
          <FlightLab />
        </Suspense>
      )}
      {tab === "Weights" && (
        <main className="secondary-main">
          <WeightVersions
            controller={gestureController}
            loaded={d?.gesture_model}
          />
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
          {(d?.model.parameters ?? d?.model.neurons ?? 0).toLocaleString()}{" "}
          {d?.model.controller === "transformer" ? "parameters" : "neurons"} ·{" "}
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
