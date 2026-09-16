import { useEffect, useState } from "react";
import { Activity, Square } from "lucide-react";
import Select from "./Select";
import type { Command, Simulation } from "../lib/types";

export function LocomotionControls({
  simulation,
  command,
  disabled,
}: {
  simulation: Simulation;
  command: Command;
  disabled: boolean;
}) {
  const [target, setTarget] = useState("DNg100");
  const [strength, setStrength] = useState(1.2);
  const walking = simulation.locomotion;
  const active = (walking?.remaining_s ?? 0) > 0;
  useEffect(() => {
    if (active && walking) setTarget(walking.target);
  }, [active, walking?.target]);
  if (!walking) return null;
  const available = walking.targets.some(
    (t) => t.type === target && t.body_ids.length > 0,
  );
  const feedback =
    simulation.senses.settings.proprioception_model === "feco-opponent-v1";
  return (
    <section className="reward-panel" aria-label="Walking circuit">
      <div className="reward-toolbar">
        <div className="reward-heading">
          <strong>Walking circuit</strong>
          <span>
            {active
              ? `${walking.target} · ${walking.remaining_s.toFixed(2)} s left${simulation.running ? "" : " · Run or Step to deliver"}`
              : "Experimental · coordinated walking not yet validated"}
          </span>
        </div>
        <Select
          aria-label="Walking neurons"
          value={target}
          disabled={disabled || active}
          onChange={setTarget}
          options={walking.targets.map((t) => ({
            value: t.type,
            label: `${t.type} · ${t.body_ids.length} neurons`,
          }))}
        />
        <label className="reward-strength">
          Drive <output>{strength.toFixed(1)}×</output>
          <input
            aria-label="Walking drive strength"
            type="range"
            min="0.1"
            max="3"
            step="0.1"
            value={strength}
            disabled={disabled || active}
            onChange={(e) => setStrength(Number(e.target.value))}
          />
        </label>
        <div className="reward-buttons">
          <button
            disabled={disabled || !available || active}
            onClick={() =>
              void command("walk", {
                target,
                amplitude: strength,
                duration: 3,
                episode: simulation.episode,
              })
            }
          >
            <Activity size={16} /> Stimulate for 3 s
          </button>
          <button
            disabled={disabled || !active}
            onClick={() =>
              void command("walk_stop", { episode: simulation.episode })
            }
          >
            <Square size={14} /> Stop drive
          </button>
        </div>
      </div>
      <details className="reward-details">
        <summary>
          {feedback
            ? "Opposing leg position and movement feedback"
            : "Legacy leg feedback"}{" "}
          · measured descending → VNC → motor connections
        </summary>
        <div className="reward-pathways">
          {walking.core.map((c) => (
            <div key={c.type}>
              <strong>{c.type}</strong>
              <span>
                {c.neurons} neurons · {c.mean_hz.toFixed(1)} Hz
              </span>
            </div>
          ))}
        </div>
        {!feedback && (
          <button
            disabled={disabled}
            onClick={() => void command("proprioception_upgrade")}
          >
            Enable FeCO leg feedback
          </button>
        )}
        <p>
          Tonic drive enters only the selected descending neurons. Every muscle
          command comes from motor-neuron activity through the full graph. Stop
          removes external drive; neural activity and muscle contraction can
          take time to decay.
        </p>
        <p>
          {feedback &&
            `${simulation.senses.leg_routing?.neurons ?? 0} identified leg receptors receive position or movement signals. `}
          FeCO cell types and leg identity use annotations. Opposing sensory
          tuning, muscle gains, and neuron physiology remain assumptions.
          Tipping, sliding, or twitching do not establish walking. This control
          is a circuit experiment, not a learned walking policy.
        </p>
      </details>
    </section>
  );
}
