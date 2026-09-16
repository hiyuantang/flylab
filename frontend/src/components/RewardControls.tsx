import { useState } from "react";
import { Minus, Plus, Square } from "lucide-react";
import type { Command, Simulation } from "../lib/types";

export function RewardControls({
  simulation,
  command,
  disabled,
}: {
  simulation: Simulation;
  command: Command;
  disabled: boolean;
}) {
  const [strength, setStrength] = useState(1);
  const reward = simulation.reward;
  if (!reward) return null;
  const active = reward.pulse_remaining_s > 0;
  const unavailable = disabled || !reward.available;
  const send = (score: number) =>
    void command("reward", { score, episode: simulation.episode });
  return (
    <section className="reward-panel" aria-label="Dopamine teaching">
      <div className="reward-toolbar">
        <div className="reward-heading">
          <strong>Associative learning</strong>
          <span className={reward.enabled ? "reward-on" : ""}>
            {!reward.available
              ? "Required anatomy unavailable"
              : active
                ? `${reward.last_event!.score > 0 ? "Positive" : "Negative"} pulse · ${(reward.pulse_remaining_s * 1000).toFixed(0)} ms left${simulation.running ? "" : " · Run or Step to deliver"}`
                : reward.enabled
                  ? "Teaching enabled · ready for feedback"
                  : "Learning off"}
          </span>
        </div>
        <label className="reward-strength">
          Strength <output>{strength.toFixed(2)}</output>
          <input
            aria-label="Teaching strength"
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            value={strength}
            disabled={unavailable || active}
            onChange={(event) => setStrength(Number(event.target.value))}
          />
        </label>
        <div className="reward-buttons">
          <button
            className="reward-positive"
            disabled={unavailable || active}
            onClick={() => send(strength)}
            title="Positive score stimulates measured PAM01 dopamine neurons"
          >
            <Plus size={16} /> Reward
          </button>
          <button
            disabled={unavailable || active}
            onClick={() => send(-strength)}
            title="Negative score stimulates measured PPL101 dopamine neurons"
          >
            <Minus size={16} /> Punish
          </button>
          <button
            disabled={unavailable || !reward.enabled}
            onClick={() =>
              void command("reward_stop", { episode: simulation.episode })
            }
            title="Stop dopamine learning and cancel the pulse; retain learned weights"
          >
            <Square size={14} /> Stop teaching
          </button>
        </div>
      </div>
      <details className="reward-details">
        <summary>
          Experimental dopamine model · {reward.changed_edges.toLocaleString()}{" "}
          / {reward.plastic_edges.toLocaleString()} eligible connections changed
        </summary>
        <div className="reward-pathways">
          {reward.pathways.map((pathway) => (
            <div key={pathway.valence}>
              <strong>
                {pathway.dan_type} → KCγ → {pathway.mbon_type}
              </strong>
              <span>
                {pathway.neurons} dopamine neurons · {pathway.dan_hz.toFixed(1)}{" "}
                Hz mean firing
              </span>
              <span>
                {pathway.plastic_edges.toLocaleString()} eligible connections ·{" "}
                {(100 * (1 - pathway.mean_factor)).toFixed(2)}% mean weakening
              </span>
            </div>
          ))}
        </div>
        <p>
          Pair feedback with the cue or action being simulated. Each score sends
          a 200 ms pulse in simulation time. Recent KC activity and local
          dopamine jointly weaken existing connections to different output
          neurons. Only firing above the pre-feedback baseline in the selected
          pathway drives learning, with a short dopamine decay after each pulse.
          Save state retains learning; Reset clears it.
        </p>
        <p>
          Wiring comes from MaleCNS. Endogenous dopamine effects are not
          modeled. Pulse strength, dopamine dynamics and the learning rule are
          assumptions, not measured physiology. Odors A and B currently share
          the same sensory encoding.
        </p>
      </details>
    </section>
  );
}
