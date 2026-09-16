import type { GestureId, GestureStatus, WeightVersion } from "./gestures";

export type ControllerKind = "connectome" | "transformer";
export type TrainingSetup = {
  iterations: number;
  batch_size: number;
  horizon: number;
  rank: number;
  learning_rate: number;
  seed: number;
  checkpoint: string;
  early_stopping: boolean;
  early_stopping_patience: number;
  demonstrations_per_gesture: number;
  validation_episodes: number;
  rollout_episodes: number;
  evaluation_interval: number;
  resume_from: "latest" | "best";
  chart_name: string;
  version_name: string;
  proportions: Record<GestureId, number>;
};
export type TrainingDraft = {
  setup: TrainingSetup;
  observedRun: string | null;
  completedRun: string | null;
  runSetup: TrainingSetup | null;
};

export function defaultTrainingSetup(kind: ControllerKind): TrainingSetup {
  return {
    iterations: kind === "transformer" ? 60 : 10,
    batch_size: kind === "transformer" ? 32 : 3,
    horizon: 25,
    rank: 2,
    learning_rate: kind === "transformer" ? 0.0003 : 0.01,
    seed: 42,
    checkpoint: "",
    early_stopping: true,
    early_stopping_patience: 10,
    demonstrations_per_gesture: 16,
    validation_episodes: 4,
    rollout_episodes: 4,
    evaluation_interval: 10,
    resume_from: "latest",
    chart_name: kind === "transformer" ? "Visual policy" : "Hand gestures",
    version_name: "",
    proportions: { palm: 1, fist: 1, point: 1, point_right: 1, point_both: 1 },
  };
}

export function setupFromVersion(
  kind: ControllerKind,
  version?: WeightVersion,
): TrainingSetup {
  const defaults = defaultTrainingSetup(kind);
  if (!version) return defaults;
  // Older transformer objectives have different units and batch semantics.
  const sameRecipe =
    kind !== "transformer" || version.training_recipe === "action-chunk-bc-v1";
  return {
    ...defaults,
    checkpoint: version.id,
    chart_name: version.chart_name,
    iterations: sameRecipe ? version.iterations : defaults.iterations,
    batch_size: sameRecipe ? version.batch_size : defaults.batch_size,
    learning_rate: sameRecipe ? version.learning_rate : defaults.learning_rate,
    horizon: version.horizon,
    rank: kind === "connectome" ? version.rank : defaults.rank,
    seed: version.seed,
    early_stopping: version.early_stopping ?? defaults.early_stopping,
    early_stopping_patience:
      version.early_stopping_patience ?? defaults.early_stopping_patience,
    demonstrations_per_gesture:
      version.demonstrations_per_gesture ?? defaults.demonstrations_per_gesture,
    validation_episodes:
      version.validation_episodes ?? defaults.validation_episodes,
    rollout_episodes: version.rollout_episodes ?? defaults.rollout_episodes,
    evaluation_interval:
      version.evaluation_interval ?? defaults.evaluation_interval,
    proportions: {
      palm: version.proportions.palm ?? 0,
      fist: version.proportions.fist ?? 0,
      point: version.proportions.point ?? 0,
      point_right: version.proportions.point_right ?? 0,
      point_both: version.proportions.point_both ?? 0,
    },
  };
}

export function sameTrainingSetup(a: TrainingSetup, b: TrainingSetup): boolean {
  return (Object.keys(a) as (keyof TrainingSetup)[]).every((key) =>
    key === "proportions"
      ? Object.entries(a.proportions).every(
          ([cue, value]) => b.proportions[cue as GestureId] === value,
        )
      : a[key] === b[key],
  );
}

export function hasCompletedTraining(status: GestureStatus): boolean {
  return (
    !status.running &&
    !status.resumable &&
    !status.error &&
    !status.cancelled &&
    ["Training complete", "Complete;", "Stopped improving;"].some((prefix) =>
      status.phase.startsWith(prefix),
    )
  );
}

export function syncTrainingDraft(
  draft: TrainingDraft,
  status: GestureStatus | null,
  kind: ControllerKind,
): TrainingDraft {
  if (!status) return draft;
  const runId = status.run_id ?? status.checkpoint;
  if (!runId) return draft;
  let next = draft;
  if (runId !== draft.observedRun) {
    const saved = status.versions?.find(
      (version) => version.id === status.checkpoint,
    );
    const runSetup = status.training_setup
      ? {
          ...defaultTrainingSetup(kind),
          ...status.training_setup,
          checkpoint: status.training_setup.checkpoint ?? "",
          proportions: {
            palm: status.training_setup.proportions.palm ?? 0,
            fist: status.training_setup.proportions.fist ?? 0,
            point: status.training_setup.proportions.point ?? 0,
            point_right: status.training_setup.proportions.point_right ?? 0,
            point_both: status.training_setup.proportions.point_both ?? 0,
          },
        }
      : saved
        ? { ...setupFromVersion(kind, saved), checkpoint: saved.parent ?? "" }
        : null;
    next = {
      ...draft,
      observedRun: runId,
      runSetup,
      setup:
        (status.running || status.resumable || hasCompletedTraining(status)) &&
        runSetup
          ? runSetup
          : draft.setup,
    };
  }
  if (
    status.running &&
    next.runSetup &&
    !sameTrainingSetup(next.setup, next.runSetup)
  ) {
    next = { ...next, setup: next.runSetup };
  }
  if (
    hasCompletedTraining(status) &&
    status.checkpoint &&
    next.completedRun !== runId &&
    status.versions?.some((version) => version.id === status.checkpoint)
  ) {
    // Completion changes only the starting weights. User tuning stays intact.
    next = {
      ...next,
      completedRun: runId,
      setup: { ...next.setup, checkpoint: status.checkpoint },
    };
  }
  return next;
}

export function isTrainingSetup(value: unknown): value is TrainingSetup {
  if (!value || typeof value !== "object") return false;
  const setup = value as TrainingSetup;
  const defaults = defaultTrainingSetup("transformer");
  return (
    (setup.resume_from === "latest" || setup.resume_from === "best") &&
    Object.entries(defaults).every(([key, expected]) => {
      const actual = setup[key as keyof TrainingSetup];
      if (key === "proportions")
        return (
          !!actual &&
          typeof actual === "object" &&
          Object.keys(defaults.proportions).every((cue) =>
            Number.isFinite((actual as Record<string, unknown>)[cue]),
          )
        );
      return (
        typeof actual === typeof expected &&
        (typeof actual !== "number" || Number.isFinite(actual))
      );
    })
  );
}
