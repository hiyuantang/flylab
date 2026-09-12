import Select from "./Select";
import { CompoundEye } from "./CompoundEye";
import { useEffect, useState } from "react";
import type { Command, SensoryFrame, SensorySettings } from "../lib/types";

function Signals({
  label,
  values,
  names,
}: {
  label: string;
  values: number[];
  names: string[];
}) {
  return (
    <div className="sense-signals">
      <strong>{label}</strong>
      <div>
        {values.map((value, i) => (
          <label key={i}>
            <span>{names[i]}</span>
            <meter
              min={0}
              max={1}
              value={value}
              aria-label={`${label} ${names[i]}`}
            />
            <output>{value.toFixed(2)}</output>
          </label>
        ))}
      </div>
    </div>
  );
}

export function Senses({
  frame,
  command,
  disabled,
  controller,
}: {
  frame: SensoryFrame;
  command: Command;
  disabled: boolean;
  controller: string;
}) {
  const [draft, setDraft] = useState<SensorySettings>(frame.settings);
  const [channel, setChannel] = useState("R1-R6");
  const compound = frame.vision.model === "compound-retina-v1";
  useEffect(() => {
    setDraft(frame.settings);
  }, [frame.settings.vision_model]);
  const update = <K extends keyof SensorySettings>(
    key: K,
    value: SensorySettings[K],
  ) => setDraft((old) => ({ ...old, [key]: value }));
  const switches = [
    ["vision_enabled", "Vision"],
    ["hearing_enabled", "Hearing"],
    ["wind_enabled", "Wind / gravity"],
    ["touch_enabled", "Touch"],
    ["proprioception_enabled", "Joint position"],
  ] as const;
  const controls = [
    ["illumination", "Light level", 0, 1, 0.1],
    ["sound_amplitude", "Tone strength", 0, 1, 0.1],
    ["sound_frequency", "Tone frequency (Hz)", 50, 1000, 10],
    ["wind_speed", "Wind speed (mm/s)", 0, 100, 5],
    ["wind_direction", "Wind direction (°)", -180, 180, 15],
  ] as const;
  const valid =
    controls.every(
      ([key, , min, max]) =>
        Number.isFinite(draft[key]) && draft[key] >= min && draft[key] <= max,
    ) &&
    draft.stimulus_position.every(
      (v, i) =>
        Number.isFinite(v) &&
        (i === 2 ? v >= 0.7 && v <= 5000 : Math.abs(v) <= 10000),
    );
  return (
    <section className="panel senses-panel">
      <div className="panel-heading">
        <div>
          <h2>What the fly senses</h2>
          <p>Live inputs at the current pose · {frame.time.toFixed(2)} s</p>
        </div>
        <span className="model-tag">Experimental sensors</span>
      </div>
      <div className="senses-readout">
        <div>
          {compound && (
            <label className="retina-channel">
              Visible-band response{" "}
              <Select
                aria-label="Retinal response channel"
                value={channel}
                onChange={(value) => setChannel(value)}
                options={[
                  { value: "R1-R6", label: "R1–R6 · broad visible" },
                  { value: "R8p", label: "R8p · blue proxy" },
                  { value: "R8y", label: "R8y · green proxy" },
                ]}
              />
            </label>
          )}
          {!compound && controller === "connectome" && (
            <button
              disabled={disabled}
              onClick={() => command("vision_upgrade")}
            >
              Use compound eyes · preserve neural state
            </button>
          )}
          <div className="eye-pair">
            {compound
              ? frame.vision.eyes.map((eye, side) => (
                  <CompoundEye
                    key={side}
                    eye={eye}
                    side={side}
                    channel={channel}
                  />
                ))
              : frame.vision.pixels.map((rows, eye) => (
                  <figure key={eye}>
                    <figcaption>
                      {eye === 0 ? "Left" : "Right"} eye{" "}
                      <span>
                        {frame.settings.vision_enabled
                          ? "16 × 8 samples"
                          : "disabled"}
                      </span>
                    </figcaption>
                    <svg
                      viewBox={`0 0 ${frame.vision.width} ${frame.vision.height}`}
                      role="img"
                      aria-label={`${eye === 0 ? "Left" : "Right"} eye grayscale view`}
                      shapeRendering="crispEdges"
                    >
                      {rows.flatMap((row, y) =>
                        row.map((value, x) => (
                          <rect
                            key={`${x}-${y}`}
                            x={x}
                            y={y}
                            width={1}
                            height={1}
                            fill={`rgb(${Math.round(value * 255)} ${Math.round(value * 255)} ${Math.round(value * 255)})`}
                          />
                        )),
                      )}
                    </svg>
                    <small>
                      Brightness {frame.vision.mean[eye].toFixed(2)}
                    </small>
                  </figure>
                ))}
          </div>
          <p className="body-copy">
            {frame.scene_id === "lab"
              ? "Head-mounted rays sample the floor, grid and dark virtual cue (no cue collider)."
              : "Head-mounted rays sample the same furniture, plants and surfaces used by collision physics."}{" "}
            {compound
              ? "Measured eye directions from Zhao et al. (2025); assumed head alignment and optical acceptance width. Dots show angular samples, not separate camera images. RGB materials provide visible-band proxies; UV, polarization and ocelli are not simulated."
              : "Eye optics, color and self-occlusion are simplified."}
          </p>
        </div>
        <div className="sensory-meters">
          <Signals label="Odor" values={frame.odor} names={["L", "R"]} />
          <Signals
            label="Antennal sound"
            values={frame.hearing}
            names={["L", "R"]}
          />
          <Signals
            label="Wind / gravity"
            values={frame.wind}
            names={["L", "R"]}
          />
          <Signals
            label="Foot contact"
            values={frame.touch}
            names={["LF", "LM", "LH", "RF", "RM", "RH"]}
          />
          <Signals
            label="Joint position"
            values={frame.proprioception}
            names={["LF", "LM", "LH", "RF", "RM", "RH"]}
          />
        </div>
      </div>
      <p className="body-copy">
        {controller === "connectome"
          ? compound
            ? `${frame.retinal_routing?.mapped ?? 0} visual neurons receive individual facet signals; ${frame.retinal_routing?.unresolved ?? 0} have unresolved spatial or spectral input and still run in the brain. Columns are inferred from measured connections. Cross-specimen column-to-eye registration is unvalidated. No eye-wide brightness pooling.`
            : `${frame.neural_routing?.visual ?? 0} visual, ${frame.neural_routing?.auditory ?? 0} auditory and ${frame.neural_routing?.wind_gravity ?? 0} wind/gravity neurons receive annotation-based inputs. Vision pools brightness per eye; retinal columns are not mapped. All tuning is assumed.`
          : controller === "posture"
            ? "Sensors are observable here, but the posture baseline uses joint feedback rather than a neural controller."
            : "The synthetic circuit receives eye brightness and antennal signals through an engineering encoder. Joint feedback also supports its gait controller."}
      </p>
      <details className="sense-settings">
        <summary>Configure sensory environment</summary>
        <fieldset disabled={disabled}>
          <legend>Sensory channels</legend>
          <div className="sense-switches">
            {switches.map(([key, label]) => (
              <label key={key}>
                <input
                  type="checkbox"
                  checked={draft[key]}
                  onChange={(e) => update(key, e.target.checked)}
                />
                {label}
              </label>
            ))}
          </div>
          <div className="lab-form">
            {controls.map(([key, label, min, max, step]) => (
              <label key={key}>
                {label}
                <input
                  aria-label={label}
                  type="number"
                  value={Number.isFinite(draft[key]) ? draft[key] : ""}
                  min={min}
                  max={max}
                  step={step}
                  onChange={(e) => update(key, e.target.valueAsNumber)}
                />
              </label>
            ))}
            {["X", "Y", "Height"].map((axis, i) => (
              <label key={axis}>
                Source {axis} (mm)
                <input
                  aria-label={`Source ${axis} (mm)`}
                  type="number"
                  value={
                    Number.isFinite(draft.stimulus_position[i])
                      ? draft.stimulus_position[i]
                      : ""
                  }
                  min={i === 2 ? 0.7 : -10000}
                  max={i === 2 ? 5000 : 10000}
                  step={0.5}
                  onChange={(e) => {
                    const p = [...draft.stimulus_position] as [
                      number,
                      number,
                      number,
                    ];
                    p[i] = e.target.valueAsNumber;
                    update("stimulus_position", p);
                  }}
                />
              </label>
            ))}
          </div>
          <p className="body-copy">
            {frame.scene_id === "lab"
              ? "The dark cue also marks the virtual sound source."
              : "Source coordinates locate a virtual tone in this scene."}{" "}
            Sound uses an assumed frequency-tuned amplitude envelope; wind is an
            antennal deflection proxy. Apply resets the experiment. Physical
            training captures these settings; odor controls remain above the
            arena.
          </p>
          <button
            disabled={!valid}
            onClick={() => void command("sensing", { senses: draft })}
          >
            Apply senses & reset
          </button>
        </fieldset>
      </details>
    </section>
  );
}
