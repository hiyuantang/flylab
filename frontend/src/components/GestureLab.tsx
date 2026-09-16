import NumberInput from "./NumberInput";
import { useViewState, oneOf } from "../lib/viewState";
import { lazy, Suspense, type ReactNode, useEffect, useState } from "react";
import {
  Play,
  Square,
  Download,
  SlidersHorizontal,
  LoaderCircle,
} from "lucide-react";
import { downloadJSON, request } from "../lib/api";
import type { Simulation } from "../lib/types";
import {
  GESTURES,
  type GestureController,
  type GestureStatus,
} from "../lib/gestures";
import { GestureIcon } from "./GestureControls";
import Select from "./Select";
import { WeightPicker } from "./WeightVersions";
import { hasCompletedTraining } from "../lib/trainingSetup";
import { CompoundEye } from "./CompoundEye";
import "./GestureLab.css";
const FlyScene = lazy(() =>
  import("./Scene").then((x) => ({ default: x.FlyScene })),
);

function LossChart({
  points,
  baseline,
  reachability,
  actionMAE = false,
}: {
  points: GestureStatus["history"];
  baseline?: number;
  actionMAE?: boolean;
  reachability?: GestureStatus["target_reachability"];
}) {
  const validation =
    baseline !== undefined &&
    points.every((p) => p.validation_loss !== undefined);
  const values = points.map((p) => ({
    ...p,
    value: validation ? p.validation_loss! : p.loss,
  }));
  if (validation)
    values.unshift({
      iteration: 0,
      loss: baseline,
      value: baseline,
      gradient_norm: 0,
      sample_losses: [],
    });
  const low = Math.min(...values.map((p) => p.value), Infinity);
  const high = Math.max(...values.map((p) => p.value), 0);
  const padding = Math.max((high - low) * 0.15, high * 0.0001, 0.00000001);
  const min = Math.max(0, Number.isFinite(low) ? low - padding : 0);
  const max = high + padding;
  const coordinates = values.map(
    (p, i) =>
      `${15 + (i / Math.max(1, values.length - 1)) * 570},${170 - ((p.value - min) / (max - min)) * 160}`,
  );
  const improvement =
    validation && baseline > 0 && values.length > 1
      ? (100 * (baseline - values.at(-1)!.value)) / baseline
      : null;
  return (
    <div className="gesture-loss">
      <div className="loss-heading">
        <h3>
          {actionMAE
            ? "Held-out action error"
            : validation
              ? "Fixed validation loss"
              : "Mean batch loss"}
        </h3>
        <span>
          {validation && `${improvement?.toFixed(2) ?? "0.00"}% improvement · `}
          {values.length > 0 &&
            `${actionMAE ? "MAE" : "MSE"} ${min.toPrecision(3)}–${max.toPrecision(3)} · ${actionMAE ? "Epoch" : "Iteration"} ${values[0].iteration}–${values.at(-1)?.iteration}`}
        </span>
      </div>
      {values.length ? (
        <>
          <svg
            viewBox="0 0 600 180"
            preserveAspectRatio="none"
            role="img"
            aria-label={`${actionMAE ? "Held-out action MAE" : "Muscle activation MSE"} over ${points.length} ${actionMAE ? "epochs" : "iterations"}`}
          >
            {[10, 90, 170].map((y) => (
              <line key={y} x1="15" x2="585" y1={y} y2={y} stroke="#30404c" />
            ))}
            <polyline
              points={coordinates.join(" ")}
              fill="none"
              stroke="#d6ee9b"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
            {coordinates.map((xy, i) => (
              <circle
                key={i}
                cx={xy.split(",")[0]}
                cy={xy.split(",")[1]}
                r="1"
                fill={
                  values[i].update_accepted === false ? "#ff9f67" : "#d6ee9b"
                }
              >
                <title>{`${actionMAE ? "Epoch" : "Iteration"} ${values[i].iteration}: ${values[i].value.toPrecision(7)}${values[i].update_accepted === false ? " · update rejected" : ""}`}</title>
              </circle>
            ))}
          </svg>
        </>
      ) : (
        <div className="gesture-empty">
          The loss curve appears after the first completed batch.
        </div>
      )}
      {!!(
        reachability?.unrouted_target_channels ??
        reachability?.unrouted_channels
      ) && (
        <p
          className="loss-reachability"
          title="These target channels have no mapped motor neurons. LoRA cannot activate them; the full target requires correcting the motor mapping."
        >
          {reachability.unrouted_channels}/{reachability.target_channels} target
          muscles have no motor route · MSE floor ≥{" "}
          {reachability.unavoidable_mse.toPrecision(4)}
        </p>
      )}
    </div>
  );
}
export function GestureLab({
  controller,
  simulation,
  legacyTools,
}: {
  controller: GestureController;
  simulation: Simulation | null;
  legacyTools?: ReactNode;
}) {
  const { status, busy, act, connectionError, kind, setKind, training } =
    controller;
  const transformer = kind === "transformer";
  const [sideTab, setSideTab] = useViewState<"setup" | "details">(
    "training.panel",
    "setup",
    oneOf(["setup", "details"]),
  );
  const [resultTab, setResultTab] = useViewState<"error" | "checks">(
    "training.resultsTab",
    "error",
    oneOf(["error", "checks"]),
  );
  const [stopPending, setStopPending] = useState(false);
  const { setup, setField, selectWeights, canResume, restorePausedSetup } =
    training;
  const {
    iterations: activeIterations,
    batch_size: activeBatchSize,
    horizon,
    rank,
    learning_rate: activeLearningRate,
    seed,
    checkpoint: parent,
    resume_from: resumeFrom,
    chart_name: chartName,
    version_name: versionName,
    proportions,
    early_stopping: earlyStopping,
    early_stopping_patience: patience,
    demonstrations_per_gesture: demonstrations,
  } = setup;
  const stopping = !status?.resumable && (stopPending || !!status?.stopping);
  const parentVersion = status?.versions?.find((v) => v.id === parent);
  const proportionTotal = Object.values(proportions).reduce((a, b) => a + b, 0);
  const disabled =
    busy ||
    stopping ||
    !!status?.running ||
    !!status?.other_training_running ||
    !status;
  const valid =
    [
      [activeIterations, 1, 1000],
      [activeBatchSize, 1, 64],
      [horizon, 5, 250],
      [seed, 0, 2147483647],
    ].every(([x, min, max]) => Number.isInteger(x) && x >= min && x <= max) &&
    (!transformer ||
      (Number.isInteger(demonstrations) &&
        demonstrations >= 1 &&
        demonstrations <= 128)) &&
    (transformer || (Number.isInteger(rank) && rank >= 1 && rank <= 64)) &&
    (!earlyStopping ||
      (Number.isInteger(patience) && patience >= 1 && patience <= 1000)) &&
    activeLearningRate > 0 &&
    activeLearningRate <= 0.1 &&
    proportionTotal > 0 &&
    Object.values(proportions).every((x) => Number.isFinite(x) && x >= 0) &&
    (parent
      ? !!parentVersion && parentVersion.compatible !== false
      : !!chartName.trim());
  const liveResults = !!status?.running || (!!status?.resumable && canResume);
  const [savedResults, setSavedResults] = useState<{
    id: string;
    results: GestureStatus;
  } | null>(null);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [checkChoice, setCheckChoice] = useState("latest");
  useEffect(() => {
    let cancelled = false;
    setSavedResults(null);
    setResultsError(null);
    setCheckChoice("latest");
    if (!liveResults && parent) {
      request<{ results: GestureStatus }>(
        `/gestures/versions/${encodeURIComponent(parent)}`,
      )
        .then(({ results }) => {
          if (!cancelled) setSavedResults({ id: parent, results });
        })
        .catch((error) => {
          if (!cancelled) setResultsError(String(error));
        });
    }
    return () => {
      cancelled = true;
    };
  }, [parent, liveResults, kind]);
  const results = liveResults
    ? status
    : savedResults?.id === parent
      ? savedResults.results
      : null;
  const actionMAE =
    (results?.loss_metric ?? status?.loss_metric) === "action_mae";
  const checks = (results?.history ?? []).filter(
    (row) => row.rollout_success_rate != null,
  );
  const checkRow =
    checkChoice === "latest"
      ? checks.at(-1)
      : checks.find((row) => String(row.iteration) === checkChoice);
  const checkEvaluation =
    checkRow?.evaluation ??
    (checkRow === checks.at(-1)
      ? results?.evaluation
      : checkRow?.iteration === results?.best_iteration
        ? results?.selected_evaluation
        : undefined);
  const previewRun = liveResults
    ? (status?.run_id ?? status?.checkpoint)
    : parent;
  const [previewEpoch, setPreviewEpoch] = useState("latest");
  const [previewSample, setPreviewSample] = useState("latest");
  const [recording, setRecording] = useState<{
    key: string;
    frame: Simulation;
  } | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const recordings = results?.previews ?? [];
  const epochs = [...new Set(recordings.map((item) => item.epoch))];
  const chosenEpoch =
    previewEpoch === "latest" ? epochs.at(-1) : Number(previewEpoch);
  const samples = recordings.filter((item) => item.epoch === chosenEpoch);
  const selectedPreview =
    previewSample === "latest"
      ? samples.at(-1)
      : samples.find((item) => item.sample === Number(previewSample));
  const previewKey =
    selectedPreview && previewRun
      ? `${previewRun}/${selectedPreview.epoch}/${selectedPreview.sample}`
      : null;
  useEffect(() => {
    setPreviewEpoch("latest");
    setPreviewSample("latest");
  }, [previewRun]);
  useEffect(() => {
    let cancelled = false;
    setRecording(null);
    setPreviewError(null);
    if (previewKey && selectedPreview && previewRun) {
      request<Simulation>(
        `/gestures/versions/${encodeURIComponent(previewRun)}/previews/${selectedPreview.epoch}/${selectedPreview.sample}`,
      )
        .then((frame) => {
          if (!cancelled) setRecording({ key: previewKey, frame });
        })
        .catch((error) => {
          if (!cancelled) setPreviewError(String(error));
        });
    }
    return () => {
      cancelled = true;
    };
  }, [previewKey]);
  const awaitingFirstBatch = !!status?.running && !status.frame && !previewKey;
  const preparingDemonstrations =
    status?.phase.startsWith("Preparing") ?? false;
  const preview = previewKey
    ? recording?.key === previewKey
      ? recording.frame
      : null
    : (status?.frame ?? (awaitingFirstBatch ? null : simulation));
  const fields = [
    {
      name: transformer ? "Epochs" : "Iterations",
      value: activeIterations,
      set: (value: number) => setField("iterations", value),
      min: 1,
      max: 1000,
    },
    {
      name: transformer ? "Batch size (windows)" : "Batch size",
      value: activeBatchSize,
      set: (value: number) => setField("batch_size", value),
      min: 1,
      max: 64,
    },
    {
      name: "Steps per sample",
      value: horizon,
      set: (value: number) => setField("horizon", value),
      min: 5,
      max: 250,
    },
    ...(transformer
      ? [
          {
            name: "Demonstrations per gesture",
            value: demonstrations,
            set: (value: number) =>
              setField("demonstrations_per_gesture", value),
            min: 1,
            max: 128,
          },
        ]
      : []),
    ...(!transformer
      ? [
          {
            name: "Adapter rank",
            value: rank,
            set: (value: number) => setField("rank", value),
            min: 1,
            max: 64,
          },
        ]
      : []),
  ];
  const comparison = status?.muscle_comparison;
  return (
    <section className="gesture-training">
      <header className="training-toolbar">
        <div className="lab-actions">
          <button
            className="primary"
            disabled={disabled || !valid}
            title={
              parent
                ? transformer &&
                  parentVersion?.training_recipe === "action-chunk-bc-v1" &&
                  parentVersion?.training_state_version
                  ? `Continue from ${resumeFrom} weights with matching Adam state; save a new child version.`
                  : "Train from the selected weights with a fresh optimizer; save a new child version."
                : undefined
            }
            onClick={() =>
              act("/gestures/start", {
                ...setup,
                checkpoint: parent || null,
              })
            }
          >
            <Play size={15} />
            {parent
              ? "Continue from version"
              : status?.resumable || status?.checkpoint
                ? "Start new"
                : transformer
                  ? "Train policy"
                  : "Train adapter"}
          </button>
          {status?.resumable && (
            <button
              className="primary"
              disabled={disabled || !canResume}
              title={
                canResume
                  ? "Resume the paused run with its unchanged setup"
                  : "Restore the paused setup to resume; changed settings start a new run"
              }
              onClick={() => act("/gestures/resume", {})}
            >
              <Play size={15} /> Resume
            </button>
          )}
          <button
            aria-busy={stopping}
            disabled={!status?.running || busy || stopping}
            onClick={async () => {
              setStopPending(true);
              try {
                await act("/gestures/stop", {});
              } finally {
                setStopPending(false);
              }
            }}
          >
            {stopping ? (
              <LoaderCircle size={15} className="training-spinner" />
            ) : (
              <Square size={15} />
            )}
            {stopping ? "Stopping…" : "Stop"}
          </button>
        </div>
        <div className="training-run-status" role="status">
          <strong>
            {stopping
              ? "Stopping · finishing current batch"
              : (status?.phase ?? "Connecting…")}
          </strong>
          <span>
            {status?.resumable
              ? canResume
                ? "Paused. Resume continues this run with its settings and optimizer."
                : "Paused run kept. The setup has changed; training starts a new run from the selected weights."
              : status?.running
                ? transformer
                  ? `${actionMAE ? "Epoch" : "Iteration"} ${status.iteration ?? 0}/${status.total} · ${status.global_step ?? 0} optimizer updates · ${status.phase}`
                  : `Iteration ${status.iteration ?? 0}/${status.total} · ${status.parallel_samples && status.parallel_samples > 1 ? `${status.parallel_samples} samples in parallel` : `sample ${status.batch_sample ?? 0}/${status.batch_size}`} · ${status.phase === "Backpropagating full sample" ? `backward ${status.backward_step ?? 0}/${status.backward_total ?? horizon}` : `forward ${status.sample_step ?? 0}/${status.sample_total ?? horizon}`}`
                : status?.checkpoint
                  ? parent === status.checkpoint && hasCompletedTraining(status)
                    ? "Saved version selected as starting weights · tuning kept"
                    : "Weights saved · load from the Experiment toolbar"
                  : "Configure a batch to begin."}
          </span>
        </div>
        <button
          className="export-training"
          disabled={!status?.history.length}
          onClick={() =>
            downloadJSON("gesture-training.json", {
              ...status,
              frame: undefined,
            })
          }
        >
          <Download size={14} />
          Export run
        </button>
        {(connectionError || status?.error) && (
          <p role="alert" className="error-inline">
            {connectionError || status?.error}
          </p>
        )}
      </header>
      <div className="training-workspace">
        <div className="training-results">
          <div className="training-preview-heading">
            <strong>
              {selectedPreview
                ? `Epoch ${selectedPreview.epoch} · ${GESTURES.find((g) => g.id === selectedPreview.gesture)?.name} · Sample ${selectedPreview.gesture_sample}`
                : status?.frame
                  ? `${actionMAE ? "Epoch" : "Iteration"} ${status.preview_iteration} · ${GESTURES.find((g) => g.id === status.preview_gesture)?.name}`
                  : awaitingFirstBatch
                    ? "Preparing first training preview"
                    : "Experiment scene · no training preview yet"}
            </strong>
            {transformer ? (
              <div className="training-preview-selectors">
                <Select
                  aria-label="Preview epoch"
                  value={previewEpoch}
                  disabled={!epochs.length}
                  onChange={(value) => {
                    setPreviewEpoch(value);
                    setPreviewSample("latest");
                  }}
                  options={[
                    {
                      value: "latest",
                      label: epochs.length
                        ? "Latest check"
                        : "No recorded previews",
                    },
                    ...epochs.map((epoch) => ({
                      value: String(epoch),
                      label: `Epoch ${epoch}`,
                    })),
                  ]}
                />
                <Select
                  aria-label="Preview sample"
                  value={previewSample}
                  disabled={!samples.length}
                  onChange={setPreviewSample}
                  options={[
                    { value: "latest", label: "Latest sample" },
                    ...samples.map((item) => ({
                      value: String(item.sample),
                      label: `${GESTURES.find((g) => g.id === item.gesture)?.name} · Sample ${item.gesture_sample}`,
                    })),
                  ]}
                />
              </div>
            ) : (
              <span>Last sample in batch · response before this update</span>
            )}
          </div>
          {preview && (
            <Suspense
              fallback={<div className="panel loading">Loading scene…</div>}
            >
              <FlyScene
                simulation={preview}
                persistenceKey="training"
                allowTargets
                targetSteps={preview?.steps ?? horizon}
              />
            </Suspense>
          )}
          {awaitingFirstBatch && (
            <div className="panel training-preview-waiting" role="status">
              <GestureIcon gesture={status?.current_gesture ?? "palm"} />
              <h3>
                {transformer
                  ? status?.phase || "Starting training"
                  : "Simulating the first batch"}
              </h3>
              <p>
                {transformer ? (
                  preparingDemonstrations ? (
                    "Preparing demonstration data before training begins. The learned response preview will appear at the first physics check."
                  ) : (
                    "The learned response preview appears during held-out physics checks every 10 epochs and at completion."
                  )
                ) : (
                  <>
                    The hands and the fly’s response appear together after all{" "}
                    {status?.batch_size ?? activeBatchSize} samples finish their
                    forward simulation. Learning continues while the preview is
                    visible.
                  </>
                )}
              </p>
              <span>
                {status?.current_gesture &&
                (!transformer || preparingDemonstrations)
                  ? `${GESTURES.find((g) => g.id === status.current_gesture)?.name} · `
                  : ""}
                {transformer
                  ? preparingDemonstrations && status?.batch_sample
                    ? `Demonstration ${status.batch_sample}/${status.sample_total ?? demonstrations} per gesture`
                    : status?.iteration
                      ? `Epoch ${status.iteration}/${status.total}`
                      : "Initializing training"
                  : status?.parallel_samples && status.parallel_samples > 1
                    ? `${status.parallel_samples} samples advancing together`
                    : `Sample ${status?.batch_sample ?? 0}/${status?.batch_size ?? activeBatchSize}`}
              </span>
            </div>
          )}
          {!preview && !awaitingFirstBatch && (
            <div className="panel loading">
              {previewError ??
                (previewKey
                  ? "Loading recorded preview…"
                  : "Waiting for the training scene…")}
            </div>
          )}
          <section className="panel training-result-panel">
            <div
              className="training-result-tabs"
              role="group"
              aria-label="Training results view"
            >
              <button
                aria-pressed={!actionMAE || resultTab === "error"}
                onClick={() => setResultTab("error")}
              >
                {actionMAE ? "Action error" : "Loss"}
              </button>
              {actionMAE && (
                <button
                  aria-pressed={resultTab === "checks"}
                  onClick={() => setResultTab("checks")}
                >
                  Gesture checks
                </button>
              )}
            </div>
            {(!actionMAE || resultTab === "error") && (
              <div
                className="training-loss-panel"
                aria-label={
                  liveResults
                    ? "Current run results"
                    : `Selected version results: ${parentVersion?.name ?? "none"}`
                }
              >
                {resultsError && (
                  <p className="training-results-source">{resultsError}</p>
                )}
                <LossChart
                  points={results?.history ?? []}
                  baseline={results?.validation_initial_loss}
                  actionMAE={actionMAE}
                  reachability={results?.target_reachability}
                />
                <div className="training-numbers">
                  {results?.validation_loss !== undefined && (
                    <span>
                      {actionMAE ? "Action MAE" : "Validation MSE"}
                      <strong>{results.validation_loss.toPrecision(5)}</strong>
                    </span>
                  )}
                  <span>
                    {actionMAE ? "Epoch loss" : "Batch loss"}
                    <strong>{results?.loss?.toPrecision(4) ?? "—"}</strong>
                  </span>
                  <span>
                    Gradient norm
                    <strong>
                      {results?.gradient_norm?.toPrecision(3) ??
                        (results?.gradient_log10_norm != null
                          ? `10^${results.gradient_log10_norm.toFixed(1)}`
                          : "—")}
                    </strong>
                  </span>
                  <span>
                    Elapsed
                    <strong>
                      {results?.wall_seconds?.toFixed(1) ?? "0.0"} s
                    </strong>
                  </span>
                </div>
              </div>
            )}
            {actionMAE && resultTab === "checks" && (
              <div className="training-physical-checks">
                <div className="physical-checks-heading">
                  <h3 title="Physics checks run every 10 epochs and at completion.">
                    Physical gesture checks
                  </h3>
                  {checkRow && (
                    <span>
                      {checkEvaluation
                        ? `${checkEvaluation.successes}/${checkEvaluation.episodes} passed`
                        : `${(100 * checkRow.rollout_success_rate!).toFixed(0)}% passed`}
                    </span>
                  )}
                  {!!checks.length && (
                    <Select
                      aria-label="Physical check history"
                      value={checkChoice}
                      onChange={setCheckChoice}
                      options={[
                        {
                          value: "latest",
                          label: `Latest check · Epoch ${checks.at(-1)!.iteration}`,
                        },
                        ...checks.map((row) => ({
                          value: String(row.iteration),
                          label: `Epoch ${row.iteration} · ${(100 * row.rollout_success_rate!).toFixed(0)}% passed`,
                        })),
                      ]}
                    />
                  )}
                </div>
                {checkRow ? (
                  <>
                    {checkEvaluation ? (
                      <div className="physical-checks-grid">
                        {Object.entries(checkEvaluation.by_gesture).map(
                          ([cue, result]) => (
                            <div className="physical-check-result" key={cue}>
                              <span>
                                {GESTURES.find((g) => g.id === cue)?.name ??
                                  cue}
                              </span>
                              <strong>
                                {result.successes}/{result.episodes}
                              </strong>
                            </div>
                          ),
                        )}
                      </div>
                    ) : (
                      <p>
                        Only the overall pass rate was saved for this check.
                      </p>
                    )}
                  </>
                ) : (
                  <p>
                    {parent || liveResults
                      ? "No physics checks recorded yet."
                      : "Select a saved version to view its checks."}
                  </p>
                )}
              </div>
            )}
          </section>
        </div>
        <aside className="panel gesture-setup-panel">
          <div className="panel-heading">
            <div>
              <h2>
                <SlidersHorizontal size={18} /> Training setup
              </h2>
              <p>
                {status?.execution?.device === "mps"
                  ? "Apple GPU"
                  : status?.execution?.device === "cuda"
                    ? "CUDA GPU"
                    : "CPU"}
                {` · ${status?.execution?.precision ?? "…"} · ${transformer ? "all weights trainable" : "frozen brain"}`}
              </p>
            </div>
            {status?.resumable && !canResume && (
              <button
                className="restore-paused-setup"
                disabled={disabled}
                onClick={restorePausedSetup}
                title="Restore the settings used by the paused run"
              >
                Restore paused setup
              </button>
            )}
          </div>
          <div
            className="training-sidebar-tabs"
            role="group"
            aria-label="Training settings view"
          >
            <button
              aria-pressed={sideTab === "setup"}
              onClick={() => setSideTab("setup")}
            >
              Tuning
            </button>
            <button
              aria-pressed={sideTab === "details"}
              onClick={() => setSideTab("details")}
            >
              Run details
            </button>
          </div>
          <div className="training-sidebar-body" hidden={sideTab !== "setup"}>
            {disabled && (
              <p className="training-setup-notice" role="status">
                {stopping
                  ? "Finishing the current batch. Setup unlocks when training pauses."
                  : status?.running
                    ? "Training in progress. Tuning is locked for this run."
                    : "Waiting for the trainer. Tuning is temporarily locked."}
              </p>
            )}
            <fieldset
              className="training-tuning"
              disabled={disabled}
              aria-label="Training setup"
            >
              <label className="policy-controller-field">
                Controller
                <Select
                  aria-label="Training controller"
                  value={kind}
                  disabled={disabled || !!status?.resumable}
                  onChange={(value) => {
                    setKind(value as typeof kind);
                  }}
                  options={[
                    {
                      value: "transformer",
                      label: "Vision–Action Transformer",
                    },
                    {
                      value: "connectome",
                      label: "MaleCNS · frozen brain + adapter",
                    },
                  ]}
                />
              </label>
              <WeightPicker
                keepVersionVisible
                baseLabel={
                  transformer
                    ? "Vanilla transformer · random weights"
                    : "Original connectome · new lineage"
                }
                versions={(status?.versions ?? []).filter(
                  (v) =>
                    v.compatible !== false &&
                    (v.controller_kind ?? "connectome") === kind,
                )}
                value={parent}
                onChange={selectWeights}
                disabled={disabled}
              />
              {transformer && (
                <>
                  <label className="policy-controller-field">
                    Continue from
                    <Select
                      aria-label="Continue from"
                      value={
                        parentVersion?.training_state_version ? resumeFrom : ""
                      }
                      onChange={(value) =>
                        setField("resume_from", value as "latest" | "best")
                      }
                      disabled={
                        disabled || !parentVersion?.training_state_version
                      }
                      options={
                        parentVersion?.training_state_version
                          ? [
                              {
                                value: "latest",
                                label: `Latest training step · ${parentVersion.global_step ?? "—"}`,
                              },
                              {
                                value: "best",
                                label: `Best saved policy · ${parentVersion.best_global_step ?? "—"}`,
                              },
                            ]
                          : [
                              {
                                value: "",
                                label: parentVersion
                                  ? "Saved weights · no optimizer history"
                                  : "From scratch · random initialization",
                              },
                            ]
                      }
                    />
                  </label>
                  <p className="hint" role="status">
                    {!parentVersion
                      ? "Starts a new weight tree with random weights and a fresh optimizer."
                      : parentVersion.training_recipe !== "action-chunk-bc-v1"
                        ? "Loads the selected weights with fresh Adam because the training objective changed to action-chunk behavior cloning."
                        : parentVersion.training_state_version
                          ? `Restores matching weights and Adam history. Uses learning rate ${activeLearningRate}. ${seed === parentVersion.seed ? "Continues the training sample stream." : "Changed seed restarts sampling and validation."}`
                          : "This older version contains weights only. Continuation starts fresh Adam and a new sample stream; future checkpoints preserve both."}
                  </p>
                </>
              )}
              <div className="training-names">
                <label>
                  Lineage name
                  <input
                    aria-label="Lineage name"
                    value={chartName}
                    maxLength={80}
                    disabled={disabled || !!parent}
                    onChange={(e) => setField("chart_name", e.target.value)}
                  />
                </label>
                <label>
                  Version name
                  <input
                    aria-label="Version name"
                    placeholder="Automatic from training settings"
                    maxLength={120}
                    value={versionName}
                    disabled={disabled}
                    onChange={(e) => setField("version_name", e.target.value)}
                  />
                </label>
              </div>
              <div className="training-fields">
                {fields.map((f) => (
                  <label key={f.name}>
                    {f.name}
                    <NumberInput
                      aria-label={f.name}

                      step="1"
                      min={f.min}
                      max={f.max}
                      value={f.value}
                      onChange={(e) => f.set(+e.target.value)}
                      disabled={
                        disabled ||
                        (f.name === "Adapter rank" && !!parentVersion)
                      }
                    />
                  </label>
                ))}
                <label>
                  Learning rate
                  <NumberInput
                    aria-label="Learning rate"

                    min="0.000001"
                    max="0.1"
                    step="0.001"
                    value={activeLearningRate}
                    onChange={(e) => setField("learning_rate", +e.target.value)}
                    disabled={disabled}
                  />
                </label>
                <label>
                  Random seed
                  <NumberInput
                    aria-label="Random seed"

                    min="0"
                    max="2147483647"
                    step="1"
                    value={seed}
                    onChange={(e) => setField("seed", +e.target.value)}
                    disabled={disabled}
                  />
                </label>
              </div>
              <div className="training-fields training-stopping-controls">
                <label>
                  Early stopping
                  <Select
                    aria-label="Early stopping"
                    value={earlyStopping ? "on" : "off"}
                    onChange={(value) =>
                      setField("early_stopping", value === "on")
                    }
                    disabled={disabled}
                    options={[
                      { value: "on", label: "Enabled" },
                      { value: "off", label: "Disabled" },
                    ]}
                  />
                </label>
                <label>
                  Patience ({transformer ? "epochs" : "iterations"})
                  <NumberInput
                    aria-label={`Patience (${transformer ? "epochs" : "iterations"})`}

                    min={1}
                    max={1000}
                    value={patience}
                    onChange={(event) =>
                      setField("early_stopping_patience", +event.target.value)
                    }
                    disabled={disabled || !earlyStopping}
                  />
                </label>
              </div>
              <div className="training-labels">
                <h3>
                  Gesture mix <span>Relative proportions</span>
                </h3>
                {GESTURES.map((g) => (
                  <div key={g.id}>
                    <GestureIcon gesture={g.id} />
                    <span>
                      {g.name}
                      <small>{g.target}</small>
                    </span>
                    <label>
                      <NumberInput
                        key={`${kind}:${parent}:${g.id}`}
                        aria-label={`${g.name} proportion`}
                        min="0"
                        step="1"
                        value={proportions[g.id]}
                        disabled={disabled}
                        onChange={(event) =>
                          setField("proportions", (p) => ({
                            ...p,
                            [g.id]: event.target.valueAsNumber,
                          }))
                        }
                      />
                      <small>
                        {proportionTotal
                          ? (
                              (100 * proportions[g.id]) /
                              proportionTotal
                            ).toFixed(0)
                          : 0}
                        %
                      </small>
                    </label>
                  </div>
                ))}
              </div>
              <p className="hint">
                Stratified mix · randomized poses · {horizon / 50} simulated
                s/sample ·{" "}
                {transformer
                  ? "One epoch visits every training window. One Adam update per minibatch. Physical checks every 10 epochs and at completion."
                  : "full-sample gradient history."}
                {!transformer &&
                  horizon < 20 &&
                  " Short trials may finish before visual signals reach the motor neurons; the measured diagnostic first reached them around 360 ms."}
              </p>
            </fieldset>
          </div>
          <div
            className="training-sidebar-body training-details-body"
            hidden={sideTab !== "details"}
          >
            {transformer && status?.optimizer_start && (
              <p className="hint" role="status">
                {status.optimizer_start}. {status.rng_start}.
                {status.parent_global_step !== undefined &&
                  ` Starting training step: ${status.parent_global_step}.`}
              </p>
            )}
            <details className="training-method">
              <summary>Training method</summary>
              {transformer ? (
                <p>
                  Behavior cloning with masked L1 error on expert action chunks.
                  Complete demonstration episodes are split into training and
                  validation sets. Each epoch shuffles training windows and Adam
                  updates all weights after each minibatch. Both eyes supply
                  eight frames; the unchanged model predicts five commands and
                  replans every 20 ms. Only eye observations enter the policy.
                  Held-out physical success selects the saved policy, with
                  action error breaking ties. Latest and best checkpoints each
                  retain matching weights, Adam, and sampling state. A changed
                  learning rate overrides the saved rate after restoring Adam.
                  Changing the training objective starts fresh Adam. Training
                  uses float32. No language model or connectome is loaded.
                </p>
              ) : (
                <>
                  <p>
                    Supervised muscle activation loss with Adam. Gradients
                    update only the constrained low-rank adapter. Motor groups
                    distinguish recorded side, leg region, and muscle identity.
                    Forward spikes remain discrete; surrogate gradients trace
                    through the entire sample, including muscle activation
                    history. GPU checkpoints every 20 ms save memory by
                    recomputing activity during backward. New runs use a
                    backward spike-derivative scale of{" "}
                    {status?.surrogate_scale_default ?? 0.0003}; the displayed
                    run used {status?.surrogate_scale ?? 1}. This scale changes
                    the training derivative, not the forward spikes. Adam keeps
                    finite updates even when validation loss increases. The best
                    weights are saved separately. Training stops after the
                    configured patience without meaningful validation
                    improvement (0.01% relative, with a 0.00000001 absolute
                    minimum). Invalid numerical updates abort the run. Fixed
                    hand inputs select weights; they do not prove generalization
                    to new scenes.
                  </p>
                  <p>
                    Every neuron, connection, and 0.1 ms neural tick is
                    retained. Apple GPU batches advance samples together using
                    shared model weights and independent neural and body states.
                    Their losses are averaged for one adapter update per
                    iteration. Apple GPU training uses float16 neural state and
                    weights, with float32 gradients, muscle loss, and adapter
                    optimizer state. This is a custom multiplicative adapter,
                    not standard additive LoRA.
                  </p>
                  <p>
                    Targets use muscle channels with mapped motor inputs. They
                    come from 50 Hz muscle commands verified in a separate
                    physics replay. All 84 main-leg channels are supervised,
                    including stance support. References that fail the pose
                    check are rejected. Training loss alone does not establish
                    reliable gesture responses.
                  </p>
                </>
              )}
            </details>
            {!!(
              status?.target_reachability?.unrouted_target_channels ??
              status?.target_reachability?.unrouted_channels
            ) && (
              <p className="hint">
                The reference commands{" "}
                {status.target_reachability.unrouted_channels} muscles without
                mapped motor inputs. Their error cannot be learned away by LoRA.
                All target channels remain in the reported MSE; no reference
                pose is claimed to be neurally reachable.
              </p>
            )}
            {status?.teacher_evaluation && (
              <p className="hint">
                Demonstration poses reached:{" "}
                {
                  Object.values(status.teacher_evaluation).filter(
                    (x) => x.pose_reached,
                  ).length
                }
                /{Object.keys(status.teacher_evaluation).length}. These are
                physically checked expert demonstrations.
              </p>
            )}
            {status?.frame?.senses?.vision.model.startsWith(
              "compound-retina-",
            ) && (
              <details className="panel training-observations">
                <summary>Last sample: visual input and muscle targets</summary>
                <div className="training-retina">
                  {status.frame.senses.vision.eyes.map((eye, i) => (
                    <CompoundEye
                      key={i}
                      eye={eye}
                      side={i}
                      channel="R1-R6"
                      model={status.frame!.senses!.vision.model}
                    />
                  ))}
                </div>
                {comparison && (
                  <div className="muscle-comparison">
                    <table>
                      <thead>
                        <tr>
                          <th>Muscle</th>
                          <th>Target activation</th>
                          <th>Actual activation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparison.names.map((name, i) => (
                          <tr key={name}>
                            <td>{name}</td>
                            <td>{comparison.target[i].toFixed(4)}</td>
                            <td>{comparison.actual[i].toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </details>
            )}
            {legacyTools}
          </div>
        </aside>
      </div>
    </section>
  );
}
