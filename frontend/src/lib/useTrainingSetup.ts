import { useEffect, useState } from "react";
import { readView, writeView } from "./viewState";
import type { GestureStatus } from "./gestures";
import {
  defaultTrainingSetup,
  isTrainingSetup,
  sameTrainingSetup,
  setupFromVersion,
  syncTrainingDraft,
  type ControllerKind,
  type TrainingDraft,
  type TrainingSetup,
} from "./trainingSetup";

function initialDraft(kind: ControllerKind): TrainingDraft {
  const defaults = defaultTrainingSetup(kind);
  const transformer = kind === "transformer";
  const previousParent = readView("training.parent", "");
  const setup = {
    ...defaults,
    iterations: readView(
      transformer ? "training.bcEpochs" : "training.iterations",
      defaults.iterations,
    ),
    batch_size: readView(
      transformer ? "training.bcBatchSize" : "training.batchSize",
      defaults.batch_size,
    ),
    learning_rate: readView(
      transformer ? "training.bcLearningRate" : "training.learningRate",
      defaults.learning_rate,
    ),
    horizon: readView("training.fullHistoryHorizon", defaults.horizon),
    rank: transformer ? 2 : readView("training.rank", defaults.rank),
    seed: readView("training.seed", defaults.seed),
    early_stopping: readView("training.earlyStopping", defaults.early_stopping),
    early_stopping_patience: readView(
      "training.patience",
      defaults.early_stopping_patience,
    ),
    demonstrations_per_gesture: readView(
      "training.bcDemonstrations",
      defaults.demonstrations_per_gesture,
    ),
    proportions: readView("training.proportions", defaults.proportions),
    checkpoint:
      previousParent.startsWith("policy-") === transformer
        ? previousParent
        : "",
    chart_name: readView("training.chartName", defaults.chart_name),
    version_name: readView("training.versionName", defaults.version_name),
  };
  const fallback = {
    setup,
    observedRun: null,
    completedRun: null,
    runSetup: null,
  };
  return readView(
    `training.draft.${kind}`,
    fallback,
    (value): value is TrainingDraft => {
      if (!value || typeof value !== "object") return false;
      const draft = value as TrainingDraft;
      return (
        isTrainingSetup(draft.setup) &&
        (draft.observedRun === null || typeof draft.observedRun === "string") &&
        (draft.completedRun === null ||
          typeof draft.completedRun === "string") &&
        (draft.runSetup === null || isTrainingSetup(draft.runSetup))
      );
    },
  );
}

export function useTrainingSetup(
  status: GestureStatus | null,
  kind: ControllerKind,
) {
  const [drafts, setDrafts] = useState(() => ({
    transformer: initialDraft("transformer"),
    connectome: initialDraft("connectome"),
  }));
  const draft = drafts[kind];
  useEffect(() => {
    setDrafts((previous) => {
      const next = syncTrainingDraft(previous[kind], status, kind);
      return next === previous[kind] ? previous : { ...previous, [kind]: next };
    });
  }, [status, kind]);
  useEffect(() => {
    writeView("training.draft.transformer", drafts.transformer);
    writeView("training.draft.connectome", drafts.connectome);
  }, [drafts]);
  const changeSetup = (update: (setup: TrainingSetup) => TrainingSetup) =>
    setDrafts((previous) => ({
      ...previous,
      [kind]: { ...previous[kind], setup: update(previous[kind].setup) },
    }));
  return {
    setup: draft.setup,
    setField: <K extends keyof TrainingSetup>(
      key: K,
      value: TrainingSetup[K] | ((old: TrainingSetup[K]) => TrainingSetup[K]),
    ) =>
      changeSetup((setup) => ({
        ...setup,
        [key]: typeof value === "function" ? value(setup[key]) : value,
      })),
    selectWeights: (id: string) => {
      const version = status?.versions?.find((item) => item.id === id);
      if (id && !version) return;
      changeSetup(() => setupFromVersion(kind, version));
    },
    canResume:
      !!status?.resumable &&
      !!draft.runSetup &&
      sameTrainingSetup(draft.setup, draft.runSetup),
    restorePausedSetup: () => {
      if (draft.runSetup) changeSetup(() => draft.runSetup!);
    },
  };
}
