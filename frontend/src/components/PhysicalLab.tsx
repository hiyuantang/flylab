import NumberInput from "./NumberInput";
import Select from "./Select";
import { useCallback, useEffect, useState } from "react";
import { Play, Square, Download, Activity } from "lucide-react";
import { request, downloadJSON } from "../lib/api";
import type { Command, SceneSummary } from "../lib/types";

type Result = {
  reward: number;
  success: boolean;
  minimum_upright: number;
  seconds: number;
  max_sampled_penetration_mm: number;
};
type Status = {
  environment?: { scene_id: string; scene_version: string };
  running: boolean;
  mode?: string;
  task?: string;
  generation?: number;
  total?: number;
  completed_rollouts?: number;
  before?: Result;
  after?: Result;
  validation?: Result[];
  checkpoint?: string;
  checkpoints: string[];
  error?: string;
  cancelled?: boolean;
  history: { generation: number; reward: number }[];
  passive_ablation?: { reward: number; success: boolean };
  reward_version?: string;
  evaluation_matches_current_dynamics?: boolean;
};
export function PhysicalLab({
  command,
  onError,
  scene,
}: {
  scene?: SceneSummary;
  command: Command;
  onError: (s: string) => void;
}) {
  const [status, setStatus] = useState<Status | null>(null);
  const [mode, setMode] = useState("connectome");
  const [task, setTask] = useState("stand");
  const [generations, setGenerations] = useState(4);
  const [horizon, setHorizon] = useState(50);
  const [busy, setBusy] = useState(false);
  const [checkpoint, setCheckpoint] = useState("");
  const [saved, setSaved] = useState<Status | null>(null);
  useEffect(() => {
    let active = true;
    setSaved(null);
    if (checkpoint)
      request<Status>(`/physical/checkpoint/${encodeURIComponent(checkpoint)}`)
        .then((x) => {
          if (active) setSaved(x);
        })
        .catch((e) => {
          if (active) onError(e.message);
        });
    return () => {
      active = false;
    };
  }, [checkpoint, onError]);
  const refresh = useCallback(
    () =>
      request<Status>("/physical")
        .then(setStatus)
        .catch((e) => onError(e.message)),
    [onError],
  );
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const value = await request<Status>("/physical");
        if (!disposed) setStatus(value);
      } catch (e) {
        if (!disposed) onError((e as Error).message);
      }
      if (!disposed) timer = setTimeout(poll, 1500);
    };
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [onError]);
  const act = async (path: string, body: unknown) => {
    setBusy(true);
    try {
      await request(path, body);
      await refresh();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="panel physical-lab">
      <div className="panel-heading">
        <div>
          <h2>Physical learning</h2>
          <p>Train against body motion, balance and muscle effort.</p>
        </div>
        <Activity size={22} />
      </div>
      <p className="body-copy">
        Training tunes six neural population gains, sensory gain and motor gain
        while preserving every imported neuron and measured connection. This is
        parameter optimization, not yet a model of biological synaptic learning.
      </p>
      <p className="body-copy">
        Next run: <strong>{scene?.title ?? "Laboratory"}</strong>
        {scene ? ` · ${scene.spawn_label}` : ""}. Captures the current scene,
        odor and sensory settings. Choose a different scene in Experiment.
        {status?.environment && (
          <span> Last run: {status.environment.scene_id}.</span>
        )}
      </p>
      <div className="lab-controls">
        <label>
          Controller
          <Select
            aria-label="Physical training controller"
            value={mode}
            onChange={(value) => setMode(value)}
            disabled={status?.running}
            options={[{ value: "connectome", label: "MaleCNS neural gains" }]}
          />
        </label>
        <label>
          Task
          <Select
            aria-label="Physical training task"
            value={task}
            onChange={(value) => setTask(value)}
            disabled={status?.running}
            options={[
              { value: "stand", label: "Maintain stance" },
              { value: "walk", label: "Reach a target" },
            ]}
          />
        </label>
        <label>
          Generations
          <NumberInput
            aria-label="Physical generations"

            min="1"
            max="100"
            value={generations}
            onChange={(e) => setGenerations(+e.target.value)}
            disabled={status?.running}
          />
        </label>
        <label>
          Steps per rollout
          <NumberInput
            aria-label="Physical horizon"

            min="5"
            max="500"
            value={horizon}
            onChange={(e) => setHorizon(+e.target.value)}
            disabled={status?.running}
          />
        </label>
      </div>
      <p className="body-copy">
        Six candidates per generation · seed 42 · {(horizon * 0.02).toFixed(2)}{" "}
        s of physics per rollout. Full-graph neural runs can take several
        minutes. Validation adds a lateral push and ±10% body mass. Stance
        success also requires height error below 0.3 mm and joint error below
        0.1 rad.
      </p>
      <div className="lab-actions">
        <button
          className="primary"
          disabled={
            busy ||
            status?.running ||
            generations < 1 ||
            generations > 100 ||
            horizon < 5 ||
            horizon > 500
          }
          onClick={() =>
            act("/physical/start", {
              mode,
              task,
              generations,
              horizon,
              population: 6,
              seed: 42,
            })
          }
        >
          <Play size={15} /> Train physical policy
        </button>
        <button
          disabled={!status?.running || busy}
          onClick={() => act("/physical/stop", {})}
        >
          <Square size={15} /> Stop
        </button>
      </div>
      <p role="status" className="body-copy">
        {status?.running
          ? `${status.mode} · generation ${status.generation}/${status.total} · ${status.completed_rollouts} rollouts finished`
          : status?.cancelled
            ? "Training stopped; incomplete runs are not applied."
            : status?.after
              ? "Training and validation complete. Review the result before applying."
              : "Ready for a physical training run."}
      </p>
      {status?.error && (
        <p role="alert" className="error-banner">
          {status.error}
        </p>
      )}
      {status?.passive_ablation && (
        <p className="body-copy">
          Zero-muscle comparison: reward{" "}
          {status.passive_ablation.reward.toFixed(3)} ·{" "}
          {status.passive_ablation.success
            ? "also meets the task criterion"
            : "does not meet the task criterion"}
          . Passive springs and contacts remain enabled.
        </p>
      )}
      {status?.after && (
        <>
          <div className="dataset-metrics">
            <div>
              <strong>{status.before?.reward.toFixed(3)}</strong>
              <span>Initial reward</span>
            </div>
            <div>
              <strong>{status.after.reward.toFixed(3)}</strong>
              <span>Trained reward</span>
            </div>
            <div>
              <strong>{status.after.success ? "Passed" : "Not reached"}</strong>
              <span>Task success criterion</span>
            </div>
            <div>
              <strong>
                {status.validation?.filter((x) => x.success).length}/
                {status.validation?.length}
              </strong>
              <span>Validation successes</span>
            </div>
          </div>
          <button
            onClick={() => downloadJSON("physical-training.json", status)}
          >
            <Download size={15} /> Export evaluation
          </button>
        </>
      )}
      <div className="lab-controls checkpoint-controls">
        <label>
          Physical checkpoint
          <Select
            aria-label="Physical checkpoint"
            value={checkpoint}
            onChange={(value) => setCheckpoint(value)}
            options={[
              { value: "", label: "Select a saved policy" },
              ...(status?.checkpoints.map((name) => ({
                value: name,
                label: name,
              })) ?? []),
            ]}
          />
        </label>
        <button
          disabled={!checkpoint || !saved || busy || status?.running}
          onClick={async () => {
            setBusy(true);
            try {
              await command("physical_load", { checkpoint });
            } finally {
              setBusy(false);
            }
          }}
        >
          Apply to experiment
        </button>
      </div>
      {saved?.after && (
        <div className="body-copy">
          <strong>Saved policy evaluation</strong>
          <p>
            {saved.environment?.scene_id ?? "lab"} · {saved.mode} · {saved.task}{" "}
            · {saved.after.seconds.toFixed(2)} s horizon ·{" "}
            {saved.reward_version ?? "legacy upright-only reward"}.
          </p>
          <p>
            Reward {saved.before?.reward.toFixed(3)} →{" "}
            {saved.after.reward.toFixed(3)}. Task criterion:{" "}
            {saved.after.success ? "passed" : "not reached"}. Validation:{" "}
            {saved.validation?.filter((x) => x.success).length}/
            {saved.validation?.length} successes.
          </p>
          {saved.evaluation_matches_current_dynamics === false && (
            <p>
              These saved results use an earlier simulation model. Re-evaluate
              this policy before comparing it with current runs.
            </p>
          )}
          <button
            onClick={() =>
              downloadJSON("saved-physical-evaluation.json", saved)
            }
          >
            Export saved evaluation
          </button>
        </div>
      )}
    </section>
  );
}
