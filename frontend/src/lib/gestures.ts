import { useViewState, oneOf } from "./viewState";
import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "./api";
import { useTrainingSetup } from "./useTrainingSetup";
import { trainingRequestGate } from "./trainingRequests";
import type { TrainingSetup } from "./trainingSetup";
import type { Simulation } from "./types";

export type GestureId =
  "palm" | "fist" | "point" | "point_right" | "point_both";
export const GESTURES: { id: GestureId; name: string; target: string }[] = [
  { id: "palm", name: "Open palms", target: "Stand" },
  { id: "fist", name: "Closed fists", target: "Crouch" },
  {
    id: "point",
    name: "Human left hand points",
    target: "Raise left front leg",
  },
  {
    id: "point_right",
    name: "Human right hand points",
    target: "Raise right front leg",
  },
  {
    id: "point_both",
    name: "Both hands point",
    target: "Raise both front legs",
  },
];
export type PolicyEvaluation = {
  trials?: { gesture: string; seed: number; success: boolean }[];
  success_rate: number;
  successes: number;
  episodes: number;
  by_gesture: Record<string, { successes: number; episodes: number }>;
};
export type GestureStatus = {
  previews?: {
    epoch: number;
    sample: number;
    gesture: GestureId;
    gesture_sample: number;
  }[];
  run_id?: string;
  training_setup?: Omit<TrainingSetup, "checkpoint"> & {
    checkpoint: string | null;
  };
  training_recipe?: string;
  loss_metric?: string;
  global_step?: number;
  evaluation?: PolicyEvaluation;
  selected_evaluation?: PolicyEvaluation;
  behavior_validated?: boolean;
  controller_kind?: "connectome" | "transformer";
  other_training_running?: boolean;
  surrogate_scale?: number;
  surrogate_scale_default?: number;
  execution?: { device: string; precision: string; gradient_precision: string };
  versions?: WeightVersion[];
  running: boolean;
  stopping?: boolean;
  resumable?: boolean;
  best_validation_loss?: number;
  best_iteration?: number;
  optimizer_start?: string;
  rng_start?: string;
  resume_from?: "latest" | "best" | "weights" | null;
  parent_global_step?: number;
  validation_continued?: boolean;
  phase: string;
  error?: string | null;
  history: {
    iteration: number;
    global_step?: number;
    rollout_success_rate?: number | null;
    evaluation?: PolicyEvaluation | null;
    loss: number;
    gradient_norm: number | null;
    gradient_log10_norm?: number | null;
    sample_losses: number[];
    validation_loss?: number;
    update_accepted?: boolean;
    update_scale?: number;
  }[];
  iteration?: number;
  total?: number;
  batch_size?: number;
  batch_sample?: number;
  parallel_samples?: number;
  batch_gestures?: GestureId[];
  sample_step?: number;
  backward_step?: number;
  backward_total?: number;
  sample_total?: number;
  preview_iteration?: number;
  evaluation_episode?: number;
  preview_gesture?: GestureId;
  current_gesture?: GestureId;
  frame?: Simulation | null;
  checkpoint?: string | null;
  checkpoints: string[];
  loaded_model?: string | null;
  wall_seconds?: number;
  loss?: number;
  gradient_norm?: number | null;
  gradient_log10_norm?: number | null;
  validation_loss?: number;
  validation_initial_loss?: number;
  update_accepted?: boolean;
  target_reachability?: {
    target_channels: number;
    unrouted_channels: number;
    unrouted_target_channels?: number;
    unavoidable_mse: number;
    unrouted_names: string[];
  };
  cancelled?: boolean;
  adapter?: { trainable_parameters: number; parameter_change: number };
  muscle_comparison?: { names: string[]; target: number[]; actual: number[] };
  teacher_evaluation?: Record<string, { pose_reached: boolean }>;
};
export function useGestures(onError: (message: string) => void) {
  const [kind, setKind] = useViewState<"connectome" | "transformer">(
    "training.controllerKind",
    "transformer",
    oneOf(["connectome", "transformer"]),
  );
  const [status, setStatus] = useState<GestureStatus | null>(null);
  const gate = useRef(trainingRequestGate());
  const actionPending = useRef(false);
  const [statusKind, setStatusKind] = useState(kind);
  const [busy, setBusy] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const accept = useCallback(
    (value: GestureStatus) =>
      setStatus((previous) => {
        if (
          value.frame &&
          previous?.frame &&
          value.run_id === previous.run_id &&
          value.checkpoint === previous.checkpoint &&
          value.preview_iteration === previous.preview_iteration &&
          value.evaluation_episode === previous.evaluation_episode &&
          value.preview_gesture === previous.preview_gesture &&
          value.frame.episode === previous.frame.episode
        )
          value.frame = previous.frame;
        return value;
      }),
    [],
  );
  useEffect(() => {
    let disposed = false;
    gate.current.invalidate();
    setStatus(null);
    setStatusKind(kind);
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      if (actionPending.current) {
        timer = setTimeout(poll, 1200);
        return;
      }
      const token = gate.current.begin();
      try {
        const value = await request<GestureStatus>(
          `/gestures?controller_kind=${kind}`,
        );
        if (!disposed && gate.current.accepts(token)) {
          accept(value);
          setConnectionError(null);
        }
      } catch (error) {
        if (!disposed && gate.current.accepts(token))
          setConnectionError((error as Error).message);
      }
      if (!disposed) timer = setTimeout(poll, 1200);
    };
    void poll();
    return () => {
      disposed = true;
      gate.current.invalidate();
      clearTimeout(timer);
    };
  }, [accept, kind]);
  const act = useCallback(
    async (path: string, body: unknown) => {
      if (actionPending.current) return false;
      actionPending.current = true;
      const token = gate.current.begin();
      setBusy(true);
      try {
        await request(`${path}?controller_kind=${kind}`, body);
        const value = await request<GestureStatus>(
          `/gestures?controller_kind=${kind}`,
        );
        if (gate.current.accepts(token)) {
          accept(value);
          setConnectionError(null);
        }
        return true;
      } catch (error) {
        if (gate.current.accepts(token)) onError((error as Error).message);
        return false;
      } finally {
        actionPending.current = false;
        setBusy(false);
      }
    },
    [accept, onError, kind],
  );
  const currentStatus = statusKind === kind ? status : null;
  const training = useTrainingSetup(currentStatus, kind);
  return {
    status: currentStatus,
    busy,
    act,
    connectionError,
    kind,
    setKind,
    training,
  };
}
export type GestureController = ReturnType<typeof useGestures>;

export type WeightVersion = {
  early_stopping?: boolean;
  early_stopping_patience?: number;
  validation_episodes?: number;
  rollout_episodes?: number;
  evaluation_interval?: number;
  training_recipe?: string;
  demonstrations_per_gesture?: number;
  selected_evaluation?: PolicyEvaluation;
  controller_kind?: "connectome" | "transformer";
  batch_mode?: "parallel" | "sequential";
  execution?: GestureStatus["execution"];
  compatible?: boolean;
  id: string;
  chart_id: string;
  chart_name: string;
  parent: string | null;
  version: number;
  name: string;
  created_at: string;
  rank: number;
  iterations: number;
  batch_size: number;
  horizon: number;
  learning_rate: number;
  seed: number;
  final_loss: number;
  wall_seconds: number;
  bytes: number;
  trainable_parameters: number;
  proportions: Record<GestureId, number>;
  optimizer_start: string;
  training_state_version?: number;
  global_step?: number;
  best_global_step?: number;
  behavior_validated: boolean;
};
