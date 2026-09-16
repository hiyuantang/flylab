import type { Simulation } from "../lib/types";

export function PerformanceMeter({
  simulation,
}: {
  simulation: Simulation | null;
}) {
  const p = simulation?.timing?.performance;
  const target =
    p?.target_hz ??
    (simulation?.timing?.muscle_command_hz ?? 50) * (simulation?.speed ?? 1);
  const budget = p?.budget_ms || 1000 / target;
  const used = p?.compute_ms ?? 0;
  const late = p?.over_budget ?? false;
  const measured = p?.achieved_hz != null;
  const status = late
    ? "Over budget · slow motion"
    : p?.phase === "waiting"
      ? "Ready · waiting for scheduled tick"
      : measured
        ? "On time"
        : "Awaiting run";
  return (
    <div
      className={`performance-meter ${late ? "late" : measured ? "on-time" : "idle"}`}
      aria-label="Simulation performance"
    >
      <span className="performance-state">
        <i />
        Performance <strong>{status}</strong>
      </span>
      <span>
        <strong>{p?.achieved_hz?.toFixed(1) ?? "—"}</strong> /{" "}
        {target.toFixed(0)} cycles/s
      </span>
      <span className="performance-budget">
        <progress
          aria-label="Cycle compute budget used"
          value={Math.min(used, budget)}
          max={budget}
        />
        {late && used < budget ? `>${budget.toFixed(1)}` : used.toFixed(1)} /{" "}
        {budget.toFixed(1)} ms
      </span>
      <span title="A missed deadline finishes its full physical step; no catch-up steps are discarded.">
        {p?.deadline_misses ?? 0} missed deadlines
        {simulation?.running ? "" : " · paused"}
      </span>
    </div>
  );
}
