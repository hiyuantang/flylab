import NumberInput from "./NumberInput";
import { lazy, Suspense, useEffect, useState } from "react";
import { Play, Pause, Wind } from "lucide-react";
import { request } from "../lib/api";
import type { Simulation } from "../lib/types";
import Select from "./Select";
import "./FlightLab.css";
const FlyScene = lazy(() =>
  import("./Scene").then((m) => ({ default: m.FlyScene })),
);
type Metrics = {
  time: number;
  position_mm: number[];
  angles_rad: number[];
  fluid_force_uN: number[];
  upright: number;
};
type Result = {
  base: Simulation;
  frames: {
    time: number;
    body: Partial<Simulation["body"]>;
    metrics: Metrics;
  }[];
  settings: { tethered: boolean };
  controller: string;
  summary: {
    mean_force_uN: number[];
    mean_torque_uN_mm: number[];
    weight_uN: number;
    lift_weight_ratio: number;
    mean_servo_power_uW: number;
    wall_seconds: number;
    simulated_seconds: number;
    warmup_seconds: number;
    averaging_cycles: number;
    averaging_interval_s: number[];
    max_tilt_degrees: number;
    max_height_error_mm: number;
    max_horizontal_drift_mm: number;
    max_wing_contacts: number;
    flight_check: "passed" | "failed" | null;
  };
};
export default function FlightLab() {
  const [tethered, setTethered] = useState(true),
    [air, setAir] = useState(true),
    [drive, setDrive] = useState(true);
  const [frequency, setFrequency] = useState(250),
    [amplitude, setAmplitude] = useState(1.3);
  const [preset, setPreset] = useState<{
    frequency: number;
    amplitude: number;
  } | null>(null);
  const [assist, setAssist] = useState(true),
    [duration, setDuration] = useState(0.04);
  const assisted = !tethered && assist;
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null),
    [frame, setFrame] = useState(0),
    [playing, setPlaying] = useState(false);
  useEffect(() => {
    let mounted = true;
    request<{ frequency: number; amplitude: number }>("/flight/configuration")
      .then((config) => {
        if (mounted) {
          setPreset(config);
          setFrequency(config.frequency);
          setAmplitude(Number(config.amplitude.toFixed(4)));
        }
      })
      .catch((e: Error) => {
        if (mounted) setError(e.message);
      });
    return () => {
      mounted = false;
    };
  }, []);
  useEffect(() => {
    if (!playing || !result) return;
    const timer = window.setInterval(
      () => setFrame((i) => (i + 1) % result.frames.length),
      25,
    );
    return () => clearInterval(timer);
  }, [playing, result]);
  async function run() {
    setBusy(true);
    setError("");
    setPlaying(false);
    try {
      const next = await request<Result>("/flight/experiment", {
        tethered,
        air,
        drive: assisted || drive,
        frequency: assisted ? preset?.frequency : frequency,
        amplitude: assisted ? preset?.amplitude : amplitude,
        controller: assisted ? "hover" : "fixed",
        duration,
      });
      setResult(next);
      setFrame(0);
      setPlaying(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const current = result?.frames[frame];
  const simulation =
    result && current
      ? {
          ...result.base,
          time: current.time,
          steps: frame,
          running: false,
          episode: frame + 1,
          body: { ...result.base.body, ...current.body },
        }
      : null;
  return (
    <main className="flight-main">
      <div className="page-heading">
        <div>
          <h1>
            <Wind size={24} /> Wing aerodynamics
          </h1>
          <p>Move the wings through air and measure the resulting forces.</p>
        </div>
      </div>
      <div className="flight-grid">
        <div>
          {simulation ? (
            <Suspense
              fallback={<div className="panel loading">Loading wing view…</div>}
            >
              <FlyScene simulation={simulation} persistenceKey="flight" />
            </Suspense>
          ) : (
            <div className="panel flight-empty">
              <Wind size={40} />
              <h2>Start with a tethered wingbeat test</h2>
              <p>
                The tether holds the body while the wing motors work against the
                air. This measures lift and torque; it is not a flight
                demonstration.
              </p>
            </div>
          )}
          {result && (
            <div className="panel flight-playback">
              <button onClick={() => setPlaying(!playing)}>
                {playing ? <Pause size={16} /> : <Play size={16} />}{" "}
                {playing ? "Pause replay" : "Play replay"}
              </button>
              <input
                aria-label="Wingbeat replay frame"
                type="range"
                min={0}
                max={result.frames.length - 1}
                value={frame}
                onChange={(e) => {
                  setPlaying(false);
                  setFrame(+e.target.value);
                }}
              />
              <span>
                {((current?.time ?? 0) * 1000).toFixed(2)} /{" "}
                {(result.summary.simulated_seconds * 1000).toFixed(0)} ms ·{" "}
                {(
                  (result.frames.length * 0.025) /
                  result.summary.simulated_seconds
                ).toFixed(0)}
                × slow motion
              </span>
              <p>
                {result.settings.tethered
                  ? "Tether attached: the body is held in place."
                  : `Free release after an 80 ms tethered spin-up. ${result.controller === "fixed-wingbeat" ? "Fixed wingbeat; no balance feedback." : "Engineering hover assist adjusts the wing commands."}`}
                {result.summary.simulated_seconds > 0.04 &&
                  " Use the 40 ms test to inspect individual wingbeats."}
              </p>
            </div>
          )}
        </div>
        <aside className="panel flight-settings">
          <h2>Physical test</h2>
          <label>
            Body constraint
            <Select
              aria-label="Flight body constraint"
              value={tethered ? "tether" : "free"}
              disabled={busy}
              onChange={(v) => {
                setTethered(v === "tether");
                setDuration(v === "tether" ? 0.04 : 1);
              }}
              options={[
                { value: "tether", label: "Tethered force measurement" },
                { value: "free", label: "Free release" },
              ]}
            />
          </label>
          {!tethered && (
            <label>
              Flight controller
              <Select
                aria-label="Flight controller"
                value={assist ? "hover" : "fixed"}
                disabled={busy}
                onChange={(v) => setAssist(v === "hover")}
                options={[
                  { value: "hover", label: "Engineering hover assist" },
                  {
                    value: "fixed",
                    label: "Fixed wingbeat · no balance feedback",
                  },
                ]}
              />
            </label>
          )}
          <label>
            Test duration
            <Select
              aria-label="Flight test duration"
              value={String(duration)}
              disabled={busy}
              onChange={(v) => setDuration(Number(v))}
              options={[
                { value: "0.04", label: "40 ms · wingbeat detail" },
                { value: "0.2", label: "200 ms · short trajectory" },
                { value: "1", label: "1 second · sustained flight check" },
              ]}
            />
          </label>
          <label>
            Air
            <Select
              aria-label="Flight air"
              value={air ? "air" : "vacuum"}
              disabled={busy}
              onChange={(v) => setAir(v === "air")}
              options={[
                { value: "air", label: "Air · 1.225 kg/m³" },
                { value: "vacuum", label: "Vacuum · no aerodynamic force" },
              ]}
            />
          </label>
          <label>
            Wing drive
            <Select
              aria-label="Wing drive"
              value={assisted || drive ? "on" : "off"}
              disabled={busy || assisted}
              onChange={(v) => setDrive(v === "on")}
              options={[
                { value: "on", label: "Flap wings" },
                { value: "off", label: "Wings stationary" },
              ]}
            />
          </label>
          <label>
            Wingbeat frequency (Hz)
            <NumberInput
              aria-label="Wingbeat frequency"

              min={150}
              max={300}
              step={10}
              value={assisted ? (preset?.frequency ?? frequency) : frequency}
              disabled={busy || !drive || assisted}
              onChange={(e) => setFrequency(+e.target.value)}
            />
          </label>
          <label>
            Stroke amplitude (radians)
            <NumberInput
              aria-label="Stroke amplitude"

              min={0.2}
              max={1.35}
              step={0.05}
              value={
                assisted
                  ? Number((preset?.amplitude ?? amplitude).toFixed(4))
                  : amplitude
              }
              disabled={busy || !drive || assisted}
              onChange={(e) => setAmplitude(+e.target.value)}
            />
          </label>
          <button
            className="primary"
            disabled={
              busy ||
              !preset ||
              (!assisted &&
                (!Number.isFinite(frequency) ||
                  !Number.isFinite(amplitude) ||
                  frequency < 150 ||
                  frequency > 300 ||
                  amplitude < 0.2 ||
                  amplitude > 1.35))
            }
            onClick={run}
          >
            <Play size={16} />
            {busy
              ? "Simulating wingbeats…"
              : `Run ${duration === 1 ? "1 second" : `${duration * 1000} ms`} test`}
          </button>
          {error && (
            <p role="alert" className="error-inline">
              {error}
            </p>
          )}
          <p className="hint">
            A one-second flight takes about 20–25 seconds to compute on this
            machine. Playback shows recorded physical motion. Hover assist uses
            calibrated settings and is not a learned gesture policy.
          </p>
          {result && (
            <section
              className="flight-measurements"
              aria-label="Aerodynamic measurements"
            >
              <h3>Measured result</h3>
              <p>
                {result.summary.averaging_cycles > 0
                  ? `Average over ${result.summary.averaging_cycles} complete wingbeats`
                  : "Average over the latter half of the test"}
              </p>
              {result.summary.flight_check && (
                <p
                  role="status"
                  className={
                    result.summary.flight_check === "passed"
                      ? "flight-check-pass"
                      : "error-inline"
                  }
                >
                  {result.summary.flight_check === "passed"
                    ? "Flight check passed for this duration"
                    : "Flight check failed for this duration"}
                </p>
              )}
              <dl>
                <dt>Upward air force</dt>
                <dd>{result.summary.mean_force_uN[2].toFixed(2)} µN</dd>
                <dt>Body weight</dt>
                <dd>{result.summary.weight_uN.toFixed(2)} µN</dd>
                <dt>Lift / weight</dt>
                <dd>{result.summary.lift_weight_ratio.toFixed(2)}</dd>
                <dt>Pitch torque about center of mass</dt>
                <dd>{result.summary.mean_torque_uN_mm[1].toFixed(2)} µN·mm</dd>
                <dt>Signed motor power</dt>
                <dd>{result.summary.mean_servo_power_uW.toFixed(1)} µW</dd>
                <dt>Compute time</dt>
                <dd>
                  {result.summary.wall_seconds.toFixed(2)} s for{" "}
                  {result.summary.simulated_seconds.toFixed(3)} simulated s
                </dd>
                {!result.settings.tethered && (
                  <>
                    <dt>Largest tilt</dt>
                    <dd>{result.summary.max_tilt_degrees.toFixed(1)}°</dd>
                    <dt>Largest height error</dt>
                    <dd>{result.summary.max_height_error_mm.toFixed(2)} mm</dd>
                    <dt>Largest horizontal drift</dt>
                    <dd>
                      {result.summary.max_horizontal_drift_mm.toFixed(2)} mm
                    </dd>
                  </>
                )}
              </dl>
              <p>
                {result.settings.tethered
                  ? "Lift above weight alone does not establish stable flight. Try a free release to test balance."
                  : "Pass requires no floor contact, tilt below 15°, height error below 1 mm and horizontal drift below 5 mm throughout the selected duration."}
              </p>
            </section>
          )}
          <details>
            <summary>Model and assumptions</summary>
            <p>
              MuJoCo models lift, drag and rotational fluid effects using an
              ellipsoid for each wing. The full articulated body remains in the
              physics simulation, running on CPU in float64. Transformer
              training remains on the GPU in float32.
            </p>
            <p>
              The wing motors are bounded engineering servos. Wing geometry,
              joint axes and fluid coefficients are approximations; wake
              interactions, flexible wings and asynchronous flight muscles are
              not resolved. This test has no trained gesture controller.
            </p>
            {result && (
              <p>
                Maximum simultaneous wing contacts:{" "}
                {result.summary.max_wing_contacts}. Wing/body collisions remain
                active and are included in the mechanics. This is not an
                experimentally validated model of living-fly aerodynamics.
              </p>
            )}
          </details>
        </aside>
      </div>
    </main>
  );
}
