import type { Command, Simulation } from "../lib/types";

export function StandingControls({
  simulation,
  command,
  disabled,
}: {
  simulation: Simulation;
  command: Command;
  disabled: boolean;
}) {
  const support = simulation.body.support;
  if (!support) return null;
  const elastic = support.elasticity_profile === "stance-elastic-v1";
  return (
    <section className="reward-panel" aria-label="Standing baseline">
      <div className="reward-toolbar">
        <div className="reward-heading">
          <strong>Standing baseline</strong>
          <span>
            {support.feet_supported
              ? "Weight supported by feet"
              : "Foot support not established"}
            {elastic
              ? " · elastic support enabled"
              : " · original joint elasticity"}
          </span>
        </div>
        <div className="reward-buttons">
          <button
            disabled={disabled}
            onClick={() => void command("standing_trial")}
            title="Back up the current state and reset into a quiet laboratory standing experiment"
          >
            Prepare standing trial
          </button>
        </div>
      </div>
      <div
        className="reward-pathways"
        aria-label="Physical support measurements"
      >
        <div>
          <strong>{support.supporting_feet} / 6 feet</strong>
          <span>carrying vertical load</span>
        </div>
        <div>
          <strong>
            {(100 * support.feet_support_fraction).toFixed(0)}% of weight
          </strong>
          <span>supported through feet</span>
        </div>
        <div>
          <strong>{support.other_vertical_uN.toFixed(2)} µN</strong>
          <span>body / upper-leg support</span>
        </div>
      </div>
      <details className="reward-details">
        <summary>Full brain · physical support · no training</summary>
        <p>
          Prepare starts a new, paused experiment; then press Run. The previous
          state is backed up. All neurons and connections run. Leg position and
          foot contact feed the brain; motor neurons drive the muscles. Vision,
          sound, wind and odor cues are off in this quiet trial.
        </p>
        <p>
          The trial uses assumed passive joint springs (20 µN·mm/rad) around the
          initial leg pose and a 0.135× firing-rate-to-muscle gain. Springs
          provide substantial support even without neural activity. There is no
          running pose tracker or external support force. This tests
          elastic-supported stance, not validated neural balance or walking.
        </p>
        <p>
          Support is instantaneous: at least three loaded feet, 80–120% of
          weight through the feet, under 5% through other body parts, and tilt
          below 30°. It does not prove sustained stability.
        </p>
      </details>
    </section>
  );
}
