import { useState } from "react";
import type { MotorMappingRecord } from "../lib/types";
import Select from "./Select";
import "./MotorEvidence.css";

const labels: Record<string, string> = {
  supported: "Supported identity",
  tentative: "Tentative identity",
  family_only: "Muscle family only",
  unresolved: "Unresolved identity",
};

export function MotorEvidence({ records }: { records: MotorMappingRecord[] }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [confidence, setConfidence] = useState("all");
  const [limit, setLimit] = useState(30);
  const filtered = records.filter((record) => {
    const identity = record.confidence?.identity;
    const text = [
      record.body_id,
      record.type,
      record.body_region,
      record.muscle_target,
      record.joint,
      record.reason,
      identity?.basis,
      ...(record.sources ?? []).map((source) => source.title),
    ].join(" ");
    return (
      (status === "all" || record.status === status) &&
      (confidence === "all" || identity?.level === confidence) &&
      text.toLowerCase().includes(query.toLowerCase())
    );
  });
  return (
    <details className="motor-evidence">
      <summary>Inspect motor sources & confidence ({records.length})</summary>
      <p className="body-copy">
        Identity confidence describes the neuron-to-muscle match. Mechanical
        confidence describes the simulated force path. All connected force paths
        are approximations; these labels are not probabilities or evidence of
        successful walking. Original reference scores use the authors’ 1–5
        scale.
      </p>
      <div className="lab-controls">
        <label>
          Search mappings
          <input
            aria-label="Search motor evidence"
            placeholder="Neuron ID, muscle, region or source"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setLimit(30);
            }}
          />
        </label>
        <label>
          Connection
          <Select
            aria-label="Motor connection status"
            value={status}
            onChange={(value) => {
              setStatus(value);
              setLimit(30);
            }}
            options={[
              { value: "all", label: "All connections" },
              { value: "mapped", label: "Connected" },
              { value: "unmapped", label: "Unresolved outputs" },
            ]}
          />
        </label>
        <label>
          Identity confidence
          <Select
            aria-label="Motor identity confidence"
            value={confidence}
            onChange={(value) => {
              setConfidence(value);
              setLimit(30);
            }}
            options={[
              { value: "all", label: "All confidence levels" },
              ...Object.entries(labels).map(([value, label]) => ({
                value,
                label,
              })),
            ]}
          />
        </label>
      </div>
      <p className="body-copy" role="status">
        {filtered.length} matching motor neurons
      </p>
      {filtered.slice(0, limit).map((record) => (
        <details key={record.body_id} className="motor-evidence-record">
          <summary>
            {record.type ?? "Unidentified"} · {record.body_id} ·{" "}
            {record.body_region}
            {" — "}
            {record.status === "mapped" ? "Connected" : "No output"}
            {" · "}
            {labels[record.confidence?.identity.level] ??
              "Evidence unavailable"}
          </summary>
          <p className="body-copy">
            <strong>Target:</strong>{" "}
            {record.muscle_target ?? record.type ?? "Unresolved"}
            {record.target_side ? ` · target side ${record.target_side}` : ""}
            {record.joint ? ` · ${record.joint}` : ""}
            {record.reason ? ` · ${record.reason.replaceAll("_", " ")}` : ""}
          </p>
          <p className="body-copy">
            <strong>Identity:</strong>{" "}
            {record.confidence?.identity.basis ??
              "Reload the backend to load evidence."}
          </p>
          <p className="body-copy">
            <strong>
              Mechanics — {record.confidence?.mechanics.level ?? "unavailable"}:
            </strong>{" "}
            {record.confidence?.mechanics.basis}
          </p>
          <ul className="motor-evidence-sources">
            {(record.sources ?? []).map((source) => (
              <li key={source.url + source.locator}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title}
                </a>
                {" · "}
                {source.locator}. {source.supports}
                {source.data_url ? (
                  <>
                    {" "}
                    <a href={source.data_url} target="_blank" rel="noreferrer">
                      Original table
                    </a>
                  </>
                ) : null}
              </li>
            ))}
          </ul>
          {(record.reference_matches ?? []).map((match, index) => (
            <p className="body-copy" key={index}>
              <strong>Reference match:</strong> {match.systematic_type} →{" "}
              {match.target || "Unspecified"}
              {" · "}
              {match["match_certainty(1-5)"]
                ? `Author score ${match["match_certainty(1-5)"]}/5`
                : "No author score"}
              {match.match_notes ? `. ${match.match_notes}` : ""}
            </p>
          ))}
        </details>
      ))}
      {filtered.length === 0 ? (
        <p className="body-copy">No mappings match these filters.</p>
      ) : null}
      {filtered.length > limit ? (
        <button onClick={() => setLimit(limit + 30)}>
          Show 30 more mappings
        </button>
      ) : null}
    </details>
  );
}
