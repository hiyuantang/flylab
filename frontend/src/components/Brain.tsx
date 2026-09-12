import Select from "./Select";
import { useState } from "react";
import { Zap, VolumeX } from "lucide-react";
import type { Simulation, Command, RegionId } from "../lib/types";
import { AnatomyExplorer } from "./AnatomyExplorer";
import { NeuralNetwork } from "./NeuralNetwork";
export function Brain({
  simulation,
  command,
  disabled,
  liveAvailable,
}: {
  simulation: Simulation;
  command: Command;
  disabled: boolean;
  liveAvailable: boolean;
}) {
  const [view, setView] = useState<"spatial" | "anatomy" | "schematic">(
    "spatial",
  );
  const [amplitude, setAmplitude] = useState(1);
  return (
    <section className="panel brain-panel">
      <div className="panel-heading">
        <div>
          <h2>Brain activity</h2>
          <p>
            {view === "spatial"
              ? "Measured soma positions · simulated neural activity"
              : view === "anatomy"
                ? "MaleCNS annotations · individual neural state"
                : simulation.model.measured_connectome
                  ? "Schematic populations · mean firing rate / 100 Hz"
                  : "Schematic regions · live neural state"}
          </p>
        </div>
        <span className="live-dot" title="Live neural state" />
      </div>
      <div className="brain-view-tabs" role="group" aria-label="Brain view">
        <button
          aria-pressed={view === "spatial"}
          onClick={() => setView("spatial")}
        >
          3D network
        </button>
        <button
          aria-pressed={view === "anatomy"}
          onClick={() => setView("anatomy")}
        >
          Anatomical groups
        </button>
        <button
          aria-pressed={view === "schematic"}
          onClick={() => setView("schematic")}
        >
          Interventions
        </button>
      </div>
      {view === "spatial" ? (
        <NeuralNetwork
          available={liveAvailable}
          stamp={`${simulation.episode}:${simulation.time}`}
        />
      ) : view === "anatomy" ? (
        <AnatomyExplorer available={liveAvailable} />
      ) : (
        <>
          <div className="region-list">
            {simulation.regions.map((r) => (
              <button
                key={r.id}
                disabled={disabled}
                onClick={() => command("select", { region: r.id })}
                className={`region-row ${r.id === simulation.selected ? "selected" : ""}`}
              >
                <span>
                  {r.name}
                  {simulation.model.measured_connectome && (
                    <small> · {r.count?.toLocaleString()}</small>
                  )}
                  {simulation.silenced.includes(r.id) && <VolumeX size={12} />}
                </span>
                <span className="bar-track">
                  <span
                    style={{ width: `${Math.min(100, r.activity * 100)}%` }}
                  />
                </span>
                <span className="mono">{r.activity.toFixed(2)}</span>
              </button>
            ))}
          </div>
          <div className="stimulation">
            <div className="field-line">
              <label htmlFor="amplitude">Stimulation strength</label>
              <span className="mono">{amplitude.toFixed(1)}</span>
            </div>
            <input
              id="amplitude"
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={amplitude}
              onChange={(e) => setAmplitude(+e.target.value)}
              disabled={disabled}
            />
            <div className="stim-controls">
              <Select
                aria-label="Stimulation region"
                value={simulation.selected}
                onChange={(value) =>
                  command("select", { region: value as RegionId })
                }
                disabled={disabled}
                options={simulation.regions.map((r) => ({
                  value: r.id,
                  label: r.name,
                }))}
              />
              <button
                className="primary"
                disabled={disabled}
                onClick={() =>
                  command("stimulate", {
                    region: simulation.selected,
                    amplitude,
                    duration: 0.5,
                  })
                }
              >
                <Zap size={16} /> Stimulate
              </button>
              <button
                className={
                  simulation.silenced.includes(simulation.selected)
                    ? "silenced"
                    : ""
                }
                title="Toggle region silencing"
                aria-label="Silence selected region"
                aria-pressed={simulation.silenced.includes(simulation.selected)}
                disabled={disabled}
                onClick={() =>
                  command("silence", {
                    region: simulation.selected,
                    enabled: !simulation.silenced.includes(simulation.selected),
                  })
                }
              >
                <VolumeX size={17} />
              </button>
            </div>
            <p className="hint">
              500 ms pulse · advances when the simulation runs
            </p>
          </div>
        </>
      )}
    </section>
  );
}
