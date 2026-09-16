import NumberInput from "./NumberInput";
import { useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  FlaskConical,
  Play,
  Square,
  Download,
} from "lucide-react";
import type {
  Training as TrainingState,
  Command,
  Evaluation,
} from "../lib/types";
import { request, downloadJSON } from "../lib/api";
import { LineChart } from "./Telemetry";
export function Training({
  training,
  command,
  disabled,
  onError,
}: {
  training: TrainingState | null;
  command: Command;
  disabled: boolean;
  onError: (s: string) => void;
}) {
  const [episodes, setEpisodes] = useState(100);
  const [reward, setReward] = useState("A");
  const [seed, setSeed] = useState(42);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!disabled)
      request<{ checkpoints: string[] }>("/training")
        .then((x) => setCheckpoints(x.checkpoints))
        .catch((e) => onError(e.message));
  }, [training?.checkpoint, disabled, onError]);
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="training-layout">
      <section className="panel training-config">
        <div className="panel-heading">
          <div>
            <h2>Teach an association</h2>
            <p>Odor conditioning</p>
          </div>
          <FlaskConical size={21} />
        </div>
        <p className="body-copy">
          Pair an odor with reinforcement. Train the circuit to approach the
          rewarded cue and avoid the other.
        </p>
        <label htmlFor="reward">Rewarded odor</label>
        <div className="segmented">
          {["A", "B"].map((x) => (
            <button
              key={x}
              id={x === "A" ? "reward" : undefined}
              className={reward === x ? "active" : ""}
              onClick={() => setReward(x)}
              disabled={!!training?.running}
            >
              Odor {x}
            </button>
          ))}
        </div>
        <label htmlFor="episodes">
          Training updates <span className="muted">32 trials per update</span>
        </label>
        <NumberInput
          id="episodes"

          min="10"
          max="1000"
          value={episodes}
          disabled={!!training?.running}
          onChange={(e) => setEpisodes(+e.target.value)}
        />
        <label htmlFor="seed">Random seed</label>
        <NumberInput
          id="seed"

          min="0"
          max="2147483647"
          value={seed}
          disabled={!!training?.running}
          onChange={(e) => setSeed(+e.target.value)}
        />
        <div className="training-actions">
          {training?.running ? (
            <button
              onClick={() => act(() => request("/training/stop", {}))}
              disabled={disabled || busy}
            >
              <Square size={15} /> Stop training
            </button>
          ) : (
            <button
              className="primary"
              disabled={
                disabled ||
                busy ||
                episodes < 10 ||
                episodes > 1000 ||
                !Number.isInteger(episodes) ||
                seed < 0 ||
                !Number.isInteger(seed)
              }
              onClick={() =>
                act(() =>
                  request("/training/start", {
                    episodes,
                    reward_odor: reward,
                    seed,
                  }),
                )
              }
            >
              <Play size={16} /> Start training
            </button>
          )}
        </div>
        <div className="explanation">
          <strong>What learns?</strong>
          <p>
            256 synaptic parameters from the modeled mushroom body to descending
            neurons. The connection pattern stays fixed.
          </p>
          <p>
            Algorithm: REINFORCE. This is a synthetic decision task, not a
            validated biological learning rule or a walking-training task.
          </p>
        </div>
      </section>
      <div className="training-results">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Learning progress</h2>
              <p>
                {training?.running
                  ? "Training a separate candidate brain"
                  : training?.after
                    ? "Run complete — ready to compare"
                    : "Your first experiment starts here"}
              </p>
            </div>
            <span className="mono">
              {training?.episodes ?? 0} / {training?.total ?? 0}
            </span>
          </div>
          <div className="progress-track">
            <div
              style={{
                width: `${((training?.episodes ?? 0) / Math.max(1, training?.total ?? 1)) * 100}%`,
              }}
            />
          </div>
          {training?.history.length ? (
            <>
              <div className="chart-legend">
                <span>
                  <i style={{ background: "#d6ee9b" }} /> Mean reward per update
                  (−1 to +1)
                </span>
              </div>
              <LineChart
                series={[training.history.map((x) => (x.reward + 1) / 2)]}
                colors={["#d6ee9b"]}
                height={180}
              />
              <div className="axis">
                <span>Update 1</span>
                <span>Reward plotted from −1 (bottom) to +1 (top)</span>
                <span>{training.episodes}</span>
              </div>
            </>
          ) : (
            <div className="empty-chart">
              <div className="empty-line" />
              <p>Rewards will appear here as the brain learns.</p>
            </div>
          )}
          <div className="result-metrics">
            <div>
              <span>Latest reward</span>
              <strong>
                {training?.history.at(-1)?.reward.toFixed(2) ?? "—"}
              </strong>
            </div>
            <div>
              <span>Parameter change · L2</span>
              <strong>
                {training?.after ? training.parameter_change.toFixed(3) : "—"}
              </strong>
            </div>
            <div>
              <span>Trial samples</span>
              <strong>
                {((training?.episodes ?? 0) * 32).toLocaleString()}
              </strong>
            </div>
          </div>
          {training?.error && (
            <p role="alert" className="error-inline">
              {training.error}
            </p>
          )}
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Before & after</h2>
            <span className="muted">
              Frozen policy · both cues · no learning
            </span>
          </div>
          <div className="evaluation-table">
            <div className="evaluation-row">
              <span>Probability of approach</span>
              <span>Before</span>
              <span>After</span>
            </div>
            {["A", "B"].map((cue, i) => (
              <div className="evaluation-row" key={cue}>
                <strong>Odor {cue}</strong>
                <span>
                  {training?.before
                    ? `${(training.before[i ? "approach_B" : "approach_A"] * 100).toFixed(1)}%`
                    : "—"}
                </span>
                <span>
                  {training?.after
                    ? `${(training.after[i ? "approach_B" : "approach_A"] * 100).toFixed(1)}%`
                    : "—"}
                </span>
              </div>
            ))}
          </div>
          <div className="inline-actions">
            <button
              className="primary"
              disabled={
                disabled ||
                !training?.after ||
                training.running ||
                training.applied
              }
              onClick={() => command("apply")}
            >
              {training?.applied ? (
                <Check size={16} />
              ) : (
                <ArrowRight size={16} />
              )}{" "}
              {training?.applied
                ? "Applied to simulation"
                : "Apply trained brain"}
            </button>
            <button
              disabled={!training?.after}
              onClick={() => downloadJSON("flylab-training.json", training)}
            >
              <Download size={15} /> Export run
            </button>
          </div>
          <p className="hint">
            Applying a brain resets body and neural state. Learned parameters
            are retained.
          </p>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <h2>Saved brains</h2>
            <button
              disabled={disabled || busy}
              onClick={() =>
                act(async () =>
                  setEvaluation(
                    await request<Evaluation>(
                      `/evaluate?reward_odor=${reward}`,
                    ),
                  ),
                )
              }
            >
              Evaluate current brain
            </button>
          </div>
          {evaluation && (
            <p className="body-copy">
              Current approach probability: A{" "}
              {(evaluation.approach_A * 100).toFixed(1)}% · B{" "}
              {(evaluation.approach_B * 100).toFixed(1)}%. These are exact
              probabilities for the two test cues, not physical navigation
              scores.
            </p>
          )}
          {checkpoints.length ? (
            <div className="checkpoint-list">
              {checkpoints.slice(0, 8).map((name) => (
                <div key={name}>
                  <span className="mono">{name}</span>
                  <button
                    disabled={disabled || !!training?.running}
                    onClick={() => command("load", { checkpoint: name })}
                  >
                    Load
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted body-copy">
              Each completed or stopped run saves a local checkpoint.
            </p>
          )}
          <button
            className="text-button"
            disabled={disabled || !!training?.running}
            onClick={() => command("restore")}
          >
            Restore untrained reference brain
          </button>
        </section>
      </div>
    </div>
  );
}
