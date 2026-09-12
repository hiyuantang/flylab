import { Download, History, Radio } from "lucide-react";
import type { Simulation } from "../lib/types";
export function LineChart({
  series,
  colors = ["#65aeff", "#ff9f67", "#dce6ed"],
  height = 110,
}: {
  series: number[][];
  colors?: string[];
  height?: number;
}) {
  const max = Math.max(1, ...series.flat().filter(Number.isFinite));
  return (
    <svg
      className="line-chart"
      viewBox={`0 0 700 ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="Recorded signal traces"
    >
      {[0, 0.5, 1].map((y) => (
        <g key={y}>
          <line
            x1="0"
            x2="700"
            y1={height - 6 - y * (height - 12)}
            y2={height - 6 - y * (height - 12)}
            stroke="#34424d"
            strokeWidth=".6"
          />
        </g>
      ))}
      {Array.from({ length: 11 }, (_, i) => (
        <line
          key={i}
          x1={i * 70}
          x2={i * 70}
          y1="0"
          y2={height}
          stroke="#34424d"
          strokeWidth=".5"
        />
      ))}
      {series.map((values, i) => (
        <polyline
          key={i}
          fill="none"
          stroke={colors[i % colors.length]}
          strokeWidth="1.7"
          vectorEffect="non-scaling-stroke"
          points={values
            .map(
              (v, j) =>
                `${(j / Math.max(1, values.length - 1)) * 700},${height - 6 - (v / max) * (height - 12)}`,
            )
            .join(" ")}
        />
      ))}
    </svg>
  );
}
export function Telemetry({
  simulation,
  history,
  onExport,
  frames,
  replay,
  onReplay,
}: {
  simulation: Simulation;
  history: Simulation["history"];
  onExport: () => void;
  frames: number;
  replay: number | null;
  onReplay: (value: number | null) => void;
}) {
  const legs = ["LF", "LM", "LH", "RF", "RM", "RH"];
  return (
    <div className="telemetry-grid">
      <section className="panel trace-panel">
        <div className="panel-heading">
          <h2>Neural → muscle → movement</h2>
          <button
            className="icon-button"
            aria-label="Export recording"
            title="Export recording"
            onClick={onExport}
            disabled={!history.length}
          >
            <Download size={16} />
          </button>
        </div>
        <div className="chart-legend">
          <span>
            <i style={{ background: "#65aeff" }} /> MB activity
          </span>
          <span>
            <i style={{ background: "#ff9f67" }} /> Muscle activation
          </span>
          <span>
            <i style={{ background: "#dce6ed" }} /> Speed (mm/s)
          </span>
        </div>
        <LineChart
          series={[
            history.map((x) => x.neural),
            history.map((x) => x.muscle),
            history.map((x) => x.speed),
          ]}
        />
        <div className="axis">
          <span>{(history[0]?.time ?? 0).toFixed(1)} s</span>
          <span>Recorded simulation time</span>
          <span>{(history.at(-1)?.time ?? 0).toFixed(1)} s</span>
        </div>
        <div className="replay-line">
          <History size={14} />
          <input
            aria-label="Replay recording"
            type="range"
            min="0"
            max={Math.max(0, frames - 1)}
            value={replay ?? Math.max(0, frames - 1)}
            disabled={frames < 2 || simulation.running}
            onChange={(e) => onReplay(+e.target.value)}
          />
          <button
            className={replay === null ? "active" : ""}
            onClick={() => onReplay(null)}
          >
            <Radio size={12} />
            {replay === null ? "Live" : "Return live"}
          </button>
        </div>
      </section>
      <section className="panel muscles-panel">
        <div className="panel-heading">
          <h2>
            Muscle activation <small>(relative)</small>
          </h2>
        </div>
        <div className="muscle-bars">
          {legs.map((leg, i) => (
            <div className="muscle-column" key={leg}>
              <span
                title={`${simulation.body.foot_forces[i].toFixed(2)} µN ground force`}
              >
                <i
                  className={`foot-state ${simulation.body.foot_contacts[i] ? "contact" : ""}`}
                />
                {leg}
              </span>
              <div className="muscle-pair">
                {[0, 1].map((j) => (
                  <div
                    className="muscle-track"
                    key={j}
                    title={`${j ? "Negative" : "Positive"} joint torque group (mean of 7 muscles): ${(simulation.body.activation[i * 2 + j] ?? 0).toFixed(3)} activation; ${(simulation.body.force[i * 2 + j] ?? 0).toFixed(5)} µN total force`}
                  >
                    <div
                      style={{
                        height: `${(simulation.body.activation[i * 2 + j] ?? 0) * 100}%`,
                        background: j ? "#ff9f67" : "#65aeff",
                      }}
                    />
                  </div>
                ))}
              </div>
              <span className="mono">
                {Math.max(
                  ...simulation.body.activation.slice(i * 2, i * 2 + 2),
                  0,
                ).toFixed(2)}
              </span>
            </div>
          ))}
        </div>
        <div className="chart-legend">
          <span>
            <i style={{ background: "#65aeff" }} /> Positive torque
          </span>
          <span>
            <i style={{ background: "#ff9f67" }} /> Negative torque
          </span>
          <span>
            {simulation.body.anatomy.muscles} effective muscles · mean by leg
          </span>
        </div>
      </section>
    </div>
  );
}
