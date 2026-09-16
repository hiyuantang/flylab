import { useViewState } from "../lib/viewState";
import { useLayoutEffect, useRef, useState } from "react";
import { GitBranch, Trash2, Download, X } from "lucide-react";
import { request, downloadJSON } from "../lib/api";
import type { GestureController, WeightVersion } from "../lib/gestures";
import "./GestureLab.css";
import Select from "./Select";

export function WeightPicker({
  versions,
  value,
  onChange,
  disabled,
  label = "Starting weights",
  baseLabel = "Original connectome · new lineage",
  keepVersionVisible = false,
}: {
  versions: WeightVersion[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
  label?: string;
  baseLabel?: string;
  keepVersionVisible?: boolean;
}) {
  const current = versions.find((v) => v.id === value);
  const charts = [...new Map(versions.map((v) => [v.chart_id, v])).values()];
  return (
    <div className="weight-picker">
      <label>
        {label}
        <Select
          aria-label={`${label} chart`}
          disabled={disabled}
          value={current?.chart_id ?? ""}
          onChange={(chartId) =>
            onChange(
              versions.filter((v) => v.chart_id === chartId).at(-1)?.id ?? "",
            )
          }
          options={[
            { value: "", label: baseLabel },
            ...charts.map((c) => ({
              value: c.chart_id,
              label: `${c.chart_name} · ${new Date(c.created_at).toLocaleString()}`,
            })),
          ]}
        />
      </label>
      {(current || keepVersionVisible) && (
        <label>
          Version
          <Select
            aria-label={`${label} version`}
            value={current ? value : ""}
            disabled={disabled || !current}
            onChange={onChange}
            options={
              current
                ? versions
                    .filter((v) => v.chart_id === current.chart_id)
                    .map((v) => ({ value: v.id, label: v.name }))
                : [{ value: "", label: "Untrained · no saved version" }]
            }
          />
        </label>
      )}
    </div>
  );
}
function descendants(versions: WeightVersion[], id: string): string[] {
  return [
    id,
    ...versions
      .filter((v) => v.parent === id)
      .flatMap((v) => descendants(versions, v.id)),
  ];
}
function Branch({
  version,
  versions,
  selected,
  onSelect,
}: {
  version: WeightVersion;
  versions: WeightVersion[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const children = versions.filter((v) => v.parent === version.id);
  return (
    <li>
      <button
        className={`weight-node ${selected === version.id ? "selected" : ""}`}
        aria-pressed={selected === version.id}
        onClick={() => onSelect(version.id)}
      >
        <span>
          Version {version.version} ·{" "}
          {version.controller_kind === "transformer"
            ? "transformer"
            : `rank ${version.rank}`}
        </span>
        <strong>{version.name}</strong>
        <small>
          {version.iterations}{" "}
          {version.training_recipe === "action-chunk-bc-v1"
            ? "epochs"
            : "iterations"}{" "}
          · batch {version.batch_size} ·{" "}
          {version.training_recipe === "action-chunk-bc-v1"
            ? "action MAE"
            : "loss"}{" "}
          {version.final_loss.toPrecision(3)}
        </small>
      </button>
      {children.length > 0 && (
        <ul>
          {children.map((v) => (
            <Branch
              key={v.id}
              version={v}
              versions={versions}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </ul>
      )}
    </li>
  );
}
export function WeightVersions({
  controller,
  loaded,
}: {
  controller: GestureController;
  loaded?: string | null;
}) {
  const { status, busy, act } = controller;
  const versions = status?.versions ?? [];
  const [selected, setSelected] = useViewState<string | null>(
    "weights.selected",
    null,
    (v): v is string | null => v === null || typeof v === "string",
  );
  const [confirm, setConfirm] = useState<{
    version: string;
    ids: string[];
    entireLineage: boolean;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const current =
    selected === null
      ? versions.at(-1)
      : versions.find((v) => v.id === selected);
  const chart = versions.filter((v) => v.chart_id === current?.chart_id);
  const lineageScroll = useRef<HTMLDivElement>(null);
  const chartId = current?.chart_id;
  useLayoutEffect(() => {
    const viewport = lineageScroll.current;
    if (viewport) {
      viewport.scrollLeft = (viewport.scrollWidth - viewport.clientWidth) / 2;
      viewport.scrollTop = 0;
    }
  }, [chartId]);
  const deletionBusy = busy || status?.running || status?.resumable;
  const affected = current ? descendants(versions, current.id) : [];
  const active = !!loaded && affected.includes(loaded);
  const exportRun = async () => {
    if (!current) return;
    try {
      const data = await request(`/gestures/versions/${current.id}`);
      downloadJSON(`${current.chart_name}-v${current.version}.json`, data);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <div className="weights-page">
      <div className="page-heading">
        <div>
          <h1>Weight versions</h1>
          <p>
            Each node is a saved weight version. Lines show which weights a
            training run started from.
          </p>
        </div>
        <span className="model-tag">Model types kept separate</span>
      </div>
      <div className="weights-layout">
        <section className="panel weight-chart">
          <div className="panel-heading">
            <h2>
              <GitBranch size={19} />
              Training lineage
            </h2>
          </div>
          <WeightPicker
            versions={versions}
            value={current?.id ?? ""}
            onChange={setSelected}
            label="Browse weights"
            disabled={busy}
          />
          <div
            ref={lineageScroll}
            className="lineage-scroll"
            tabIndex={0}
            aria-label="Training lineage tree"
          >
            <div className="lineage-canvas">
              <div className="weight-root">
                Controller initialization
                <small>Original connectome or new transformer weights</small>
              </div>
              {chart.length > 0 ? (
                <ul className="lineage-tree">
                  {chart
                    .filter((v) => !v.parent)
                    .map((v) => (
                      <Branch
                        key={v.id}
                        version={v}
                        versions={chart}
                        selected={current?.id ?? ""}
                        onSelect={setSelected}
                      />
                    ))}
                </ul>
              ) : (
                <p className="gesture-empty">
                  Your first completed training run will create a lineage here.
                </p>
              )}
            </div>
          </div>
        </section>
        <aside className="panel weight-details">
          <h2>{current?.name ?? "No saved adapters"}</h2>
          {current && (
            <>
              <p>
                {current.chart_name} · version {current.version}
              </p>
              <dl>
                <dt>Training format</dt>
                <dd>
                  {current.compatible === false
                    ? "Earlier experiment; cannot resume"
                    : "Muscle gradients"}
                </dd>
                <dt>Started from</dt>
                <dd>
                  {versions.find((v) => v.id === current.parent)?.name ??
                    (current.controller_kind === "transformer"
                      ? "Random transformer weights"
                      : "Frozen original")}
                </dd>
                <dt>Lineage id</dt>
                <dd>{current.chart_id}</dd>
                <dt>Created</dt>
                <dd>{new Date(current.created_at).toLocaleString()}</dd>
                <dt>Training</dt>
                <dd>
                  {current.iterations}{" "}
                  {current.training_recipe === "action-chunk-bc-v1"
                    ? "epochs"
                    : "iterations"}{" "}
                  · batch {current.batch_size}{" "}
                  {current.training_recipe === "action-chunk-bc-v1"
                    ? "windows"
                    : "samples"}
                </dd>
                <dt>Gesture mix</dt>
                <dd>
                  Palm {current.proportions.palm}% · fists{" "}
                  {current.proportions.fist}% · left {current.proportions.point}
                  % · right {current.proportions.point_right ?? 0}% · both{" "}
                  {current.proportions.point_both ?? 0}%
                </dd>
                <dt>
                  {current.controller_kind === "transformer"
                    ? "Trainable parameters"
                    : "Rank / parameters"}
                </dt>
                <dd>
                  {current.controller_kind !== "transformer" &&
                    `${current.rank} / `}
                  {current.trainable_parameters.toLocaleString()}
                </dd>
                <dt>Learning rate / seed</dt>
                <dd>
                  {current.learning_rate} / {current.seed}
                </dd>
                <dt>
                  {current.training_recipe === "action-chunk-bc-v1"
                    ? "Saved action MAE"
                    : "Saved model loss"}
                </dt>
                <dd>{current.final_loss.toPrecision(5)}</dd>
                <dt>Weight file</dt>
                <dd>{(current.bytes / 1024).toFixed(1)} KB</dd>
                <dt>Initialization</dt>
                <dd>{current.optimizer_start}</dd>
                <dt>Training precision</dt>
                <dd>
                  {current.execution?.device === "mps" ? "Apple GPU" : "CPU"}
                  {" · "}
                  {current.execution?.precision ?? "float64"}
                  {current.execution?.gradient_precision === "float32" &&
                    " · float32 gradients"}
                </dd>
                <dt>Batch execution</dt>
                <dd>
                  {current.training_recipe === "action-chunk-bc-v1"
                    ? "Shuffled window minibatches"
                    : current.batch_mode === "parallel"
                      ? "Parallel GPU samples"
                      : "Sequential samples"}
                </dd>
                <dt>Behavior</dt>
                <dd>
                  {current.selected_evaluation
                    ? `${current.selected_evaluation.successes}/${current.selected_evaluation.episodes} held-out trials passed`
                    : current.behavior_validated
                      ? "Validated"
                      : "Not yet validated"}
                </dd>
              </dl>
              <div className="lab-actions">
                <button
                  disabled={
                    busy || status?.running || current.compatible === false
                  }
                  onClick={() =>
                    act("/gestures/load", { checkpoint: current.id })
                  }
                >
                  Load version
                </button>
                <button onClick={exportRun}>
                  <Download size={14} />
                  Run log
                </button>
                <button
                  className="danger"
                  disabled={deletionBusy || active}
                  onClick={() =>
                    setConfirm({
                      version: current.id,
                      ids: affected,
                      entireLineage: !current.parent,
                    })
                  }
                >
                  <Trash2 size={14} />
                  {current.parent ? "Delete branch" : "Delete entire lineage"}
                </button>
              </div>
              {active && (
                <p className="hint">
                  Load the original or a different branch before deleting the
                  active model.
                </p>
              )}
              <p className="hint">
                {current.controller_kind === "transformer"
                  ? "Transformer versions store full weights, sensor/action schema, and training logs. Starting from an earlier version copies its weights into a new run with a fresh optimizer."
                  : "Connectome versions store adapter tensors and run metadata. Starting from an earlier version copies its adapter; the base stays frozen."}
              </p>
            </>
          )}
          {error && <p role="alert">{error}</p>}
        </aside>
      </div>
      {confirm && current && (
        <div className="weight-modal-backdrop">
          <div
            className="panel weight-delete-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-weights-title"
          >
            <button
              className="dialog-close"
              aria-label="Cancel deletion"
              onClick={() => setConfirm(null)}
            >
              <X size={18} />
            </button>
            <h2 id="delete-weights-title">
              {confirm.entireLineage
                ? "Delete entire lineage"
                : "Delete branch"}
              ? ({confirm.ids.length} weight{" "}
              {confirm.ids.length === 1 ? "version" : "versions"})
            </h2>
            <p>
              {confirm.entireLineage
                ? "This permanently removes every saved version in this lineage, including all branches, weight files, and logs. Controller initialization remains available for new training."
                : "This permanently removes the selected version and all its descendants, including their weight files and logs. Sibling branches are kept."}
            </p>
            <p>
              Selected version:{" "}
              <strong>
                {versions.find((v) => v.id === confirm.version)?.name ??
                  confirm.version}
              </strong>
            </p>
            <div className="lab-actions">
              <button onClick={() => setConfirm(null)}>Cancel</button>
              <button
                className="danger"
                disabled={busy}
                onClick={async () => {
                  await act("/gestures/versions/delete", {
                    version: confirm.version,
                    expected_ids: confirm.ids,
                  });
                  setConfirm(null);
                  setSelected("");
                }}
              >
                {confirm.entireLineage
                  ? "Delete entire lineage permanently"
                  : "Delete branch permanently"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
