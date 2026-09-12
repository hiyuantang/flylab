import Select from "./Select";
import { useEffect, useState } from "react";
import { request, downloadJSON } from "../lib/api";
type Mechanics = {
  parameters: {
    mass_scale: number;
    strength_scale: number;
    friction: number;
    limit_mode: string;
  };
  mass_mg: number;
  max_force_parameter_uN: number;
  pretarsal_max_force_parameter_uN: number | null;
  provenance: string;
  joints: {
    name: string;
    range_degrees: number[];
    range: number[];
    range_unit: string;
    stiffness: number;
    damping: number;
    active: boolean;
  }[];
};
type FullGraph = {
  available: boolean;
  neurons: number;
  edges: number;
  retained_synapses: number;
  boundary_connection_rows: number;
  excluded_annotation_rows: number;
  selection: string;
};
type Mapping = {
  available: boolean;
  mapping_profile: string;
  total_motor_neurons: number;
  coverage_by_region: Record<
    string,
    { total: number; mapped: number; unmapped: number }
  >;
  unmapped: {
    body_id: number;
    type: string | null;
    body_region: string;
    reason: string;
  }[];
  mapped_motor_neurons: number;
  unmapped_motor_neurons: number;
  actuated_channels: number;
  total_channels: number;
  limitations: string;
  unassigned_region_neurons: number;
  mapping: { body_id: number; type: string; leg: string; joint: string }[];
  physiology: {
    physiology: {
      gain: number;
      membrane_ms: number;
      synapse_ms: number;
      glutamate_sign: number;
    };
  };
};
type Execution = {
  current: { device: "cpu" | "mps"; precision: string; note: string } | null;
  available: { cpu: boolean; mps: boolean };
  scope: string;
};
export function ModelControls({ onError }: { onError: (s: string) => void }) {
  const [execution, setExecution] = useState<Execution | null>(null);
  const [mechanics, setMechanics] = useState<Mechanics | null>(null);
  const [graph, setGraph] = useState<FullGraph | null>(null);
  const [mapping, setMapping] = useState<Mapping | null>(null);
  const [gain, setGain] = useState(0.275);
  const [membrane, setMembrane] = useState(20);
  const [synapse, setSynapse] = useState(5);
  const [glutamate, setGlutamate] = useState(-1);
  const [busy, setBusy] = useState(false);
  const [motorQuery, setMotorQuery] = useState("");
  useEffect(() => {
    let active = true;
    Promise.all([
      request<Mechanics>("/mechanics"),
      request<FullGraph>("/connectome/full"),
      request<Mapping>("/connectome/mapping"),
      request<Execution>("/execution"),
    ])
      .then(([m, g, n, executionStatus]) => {
        if (active) {
          setMechanics(m);
          setGraph(g);
          setMapping(n);
          setExecution(executionStatus);
          if (n.available) {
            const p = n.physiology.physiology;
            setGain(p.gain);
            setMembrane(p.membrane_ms);
            setSynapse(p.synapse_ms);
            setGlutamate(p.glutamate_sign);
          }
        }
      })
      .catch((e) => onError(e.message));
    return () => {
      active = false;
    };
  }, [onError]);
  const switchDevice = async (device: "cpu" | "mps") => {
    setBusy(true);
    try {
      setExecution(await request<Execution>("/execution", { device }));
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const saveBody = async () => {
    if (!mechanics) return;
    setBusy(true);
    try {
      setMechanics(
        await request<Mechanics>("/mechanics", mechanics.parameters),
      );
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const applyMapping = async () => {
    setBusy(true);
    try {
      setMapping(
        await request<Mapping>("/connectome/mapping", {
          profile: "muscle-routing-v4",
        }),
      );
      setMechanics(await request<Mechanics>("/mechanics"));
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const saveNeural = async () => {
    setBusy(true);
    try {
      await request("/connectome/physiology", {
        gain,
        membrane_ms: membrane,
        synapse_ms: synapse,
        glutamate_sign: glutamate,
      });
      setMapping(await request<Mapping>("/connectome/mapping"));
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      {execution?.current && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Brain execution</h2>
              <p>
                {execution.current.device.toUpperCase()} ·{" "}
                {execution.current.precision.replace("torch.", "")}
              </p>
            </div>
          </div>
          <p className="body-copy">{execution.current.note}</p>
          <div className="lab-controls">
            <button
              disabled={busy || execution.current.device === "cpu"}
              onClick={() => switchDevice("cpu")}
            >
              Use CPU · float64 reference
            </button>
            <button
              disabled={
                busy ||
                !execution.available.mps ||
                execution.current.device === "mps"
              }
              onClick={() => switchDevice("mps")}
            >
              Use Apple GPU · float32 experimental
            </button>
          </div>
          <p className="body-copy">
            Switching pauses the simulation, saves a backup, and transfers every
            neuron and pending spike without resetting time. {execution.scope}
          </p>
          {!execution.available.mps && (
            <p className="body-copy">
              Apple GPU is unavailable on this server.
            </p>
          )}
        </section>
      )}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Whole annotated MaleCNS graph</h2>
            <p>Every classified neuron; all retained connections.</p>
          </div>
        </div>
        {graph?.available ? (
          <>
            <div className="dataset-metrics">
              <div>
                <strong>{graph.neurons.toLocaleString()}</strong>
                <span>Annotated neurons</span>
              </div>
              <div>
                <strong>{graph.edges.toLocaleString()}</strong>
                <span>Directed connections</span>
              </div>
              <div>
                <strong>{graph.retained_synapses.toLocaleString()}</strong>
                <span>Retained synapses</span>
              </div>
            </div>
            <p className="body-copy">
              {graph.selection}{" "}
              {graph.boundary_connection_rows.toLocaleString()} boundary rows
              connect to excluded segments.{" "}
              {graph.excluded_annotation_rows.toLocaleString()} unclassified
              annotation rows remain outside this identified-neuron graph.
            </p>
            <div className="lab-controls">
              <label>
                Efficacy per synapse (mV)
                <input
                  aria-label="Synaptic gain"
                  step="0.025"
                  type="number"
                  min="0"
                  max="100"
                  value={gain}
                  onChange={(e) => setGain(+e.target.value)}
                />
              </label>
              <label>
                Membrane decay (ms)
                <input
                  aria-label="Membrane decay"
                  type="number"
                  min="1"
                  max="100"
                  value={membrane}
                  onChange={(e) => setMembrane(+e.target.value)}
                />
              </label>
              <label>
                Synaptic decay (ms)
                <input
                  aria-label="Synaptic decay"
                  type="number"
                  min="1"
                  max="100"
                  value={synapse}
                  onChange={(e) => setSynapse(+e.target.value)}
                />
              </label>
              <label>
                Glutamate assumption
                <Select
                  aria-label="Glutamate sign"
                  value={glutamate}
                  onChange={(value) => setGlutamate(+value)}
                  options={[
                    { value: "-1", label: "Inhibitory CNS effect" },
                    { value: "0", label: "No fast effect" },
                    { value: "1", label: "Excitatory effect" },
                  ]}
                />
              </label>
              <button
                disabled={
                  busy ||
                  !Number.isFinite(gain) ||
                  gain < 0 ||
                  gain > 100 ||
                  membrane < 1 ||
                  membrane > 100 ||
                  synapse < 1 ||
                  synapse > 100
                }
                onClick={saveNeural}
              >
                Apply physiology & reset
              </button>
            </div>
          </>
        ) : (
          <p className="body-copy">
            Full graph not prepared. Use the full-graph importer described in
            README.
          </p>
        )}
        {mapping?.available && (
          <>
            <p className="body-copy">
              <strong>
                {mapping.mapped_motor_neurons}/{mapping.total_motor_neurons}{" "}
                motor neurons → {mapping.actuated_channels}/
                {mapping.total_channels} actuator channels.
              </strong>{" "}
              {mapping.limitations}
            </p>
            <p className="body-copy">
              {mapping.unmapped_motor_neurons} motor neurons across the brain
              and VNC have no supported actuator mapping.{" "}
              {mapping.unassigned_region_neurons.toLocaleString()} neurons run
              in the network but are outside the six displayed population
              groups.
            </p>
            <p className="body-copy">
              Mapping: {mapping.mapping_profile}. The peripheral upgrade adds
              named mouthpart, neck, wing and haltere muscle channels with
              effective force transmissions. Joint geometry, force scaling and
              elasticity are uncalibrated. Wing power uses a quasi-static
              approximation; asynchronous flight and fluid ingestion are not
              simulated.
            </p>
            {mapping.mapping_profile !== "muscle-routing-v4" && (
              <button disabled={busy} onClick={applyMapping}>
                Add named peripheral muscles · preserve neural state
              </button>
            )}
            <div className="lab-table-scroll">
              <table aria-label="Motor connection coverage">
                <thead>
                  <tr>
                    <th>Body region</th>
                    <th>Connected</th>
                    <th>Unresolved</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(mapping.coverage_by_region).map(
                    ([region, counts]) => (
                      <tr key={region}>
                        <td>{region}</td>
                        <td>
                          {counts.mapped}/{counts.total}
                        </td>
                        <td>{counts.unmapped}</td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
            <details>
              <summary>
                Inspect missing muscle connections (
                {mapping.unmapped_motor_neurons})
              </summary>
              <div className="anatomy-search">
                <input
                  aria-label="Search missing muscle connections"
                  placeholder="Neuron ID, target, region or reason"
                  value={motorQuery}
                  onChange={(e) => setMotorQuery(e.target.value)}
                />
              </div>
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Neuron ID</th>
                      <th>Annotated target</th>
                      <th>Region</th>
                      <th>Missing evidence/mechanism</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mapping.unmapped
                      .filter((m) =>
                        `${m.body_id} ${m.type ?? ""} ${m.body_region} ${m.reason.replaceAll("_", " ")}`
                          .toLowerCase()
                          .includes(motorQuery.toLowerCase()),
                      )
                      .map((m) => (
                        <tr key={m.body_id}>
                          <td>{m.body_id}</td>
                          <td>{m.type ?? "Unidentified"}</td>
                          <td>{m.body_region}</td>
                          <td>{m.reason.replaceAll("_", " ")}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </details>
            <details>
              <summary>Inspect annotated muscle targets</summary>
              <div className="lab-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Neuron ID</th>
                      <th>Annotated target</th>
                      <th>Model joint</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mapping.mapping.map((m) => (
                      <tr key={m.body_id}>
                        <td>{m.body_id}</td>
                        <td>{m.type}</td>
                        <td>{m.joint}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
            <button
              onClick={() =>
                downloadJSON("malecns-muscle-mapping.json", mapping)
              }
            >
              Export mapping & assumptions
            </button>
          </>
        )}
      </section>
      {mechanics && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Body mechanics</h2>
              <p>
                {mechanics.mass_mg.toFixed(3)} mg ·{" "}
                {mechanics.max_force_parameter_uN.toFixed(0)} µN maximum force
                parameter per leg-joint actuator
                {mechanics.pretarsal_max_force_parameter_uN != null &&
                  ` · ${mechanics.pretarsal_max_force_parameter_uN.toFixed(0)} µN per pretarsal tendon`}
              </p>
            </div>
          </div>
          <p className="body-copy">{mechanics.provenance}</p>
          <div className="lab-controls">
            {(["mass_scale", "strength_scale", "friction"] as const).map(
              (key) => (
                <label key={key}>
                  {
                    {
                      mass_scale: "Mass multiplier",
                      strength_scale: "Muscle strength multiplier",
                      friction: "Ground friction",
                    }[key]
                  }
                  <input
                    aria-label={key}
                    type="number"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={mechanics.parameters[key]}
                    onChange={(e) =>
                      setMechanics({
                        ...mechanics,
                        parameters: {
                          ...mechanics.parameters,
                          [key]: +e.target.value,
                        },
                      })
                    }
                  />
                </label>
              ),
            )}
            <label>
              Joint ranges
              <Select
                aria-label="Joint range profile"
                value={mechanics.parameters.limit_mode}
                onChange={(value) =>
                  setMechanics({
                    ...mechanics,
                    parameters: {
                      ...mechanics.parameters,
                      limit_mode: value,
                    },
                  })
                }
                options={[
                  { value: "baseline", label: "Assumed baseline limits" },
                  {
                    value: "reference_envelope",
                    label: "Recorded gait envelope + margin",
                  },
                ]}
              />
            </label>
          </div>
          <button disabled={busy} onClick={saveBody}>
            Apply mechanics & reset
          </button>
          <details>
            <summary>Inspect all joint limits, stiffness and damping</summary>
            <div className="lab-table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Joint</th>
                    <th>Range</th>
                    <th>Stiffness</th>
                    <th>Damping</th>
                    <th>Control</th>
                  </tr>
                </thead>
                <tbody>
                  {mechanics.joints.map((j) => (
                    <tr key={j.name}>
                      <td>{j.name}</td>
                      <td>
                        {j.range
                          .map((x) => x.toFixed(j.range_unit === "mm" ? 3 : 1))
                          .join(" to ")}{" "}
                        {j.range_unit}
                      </td>
                      <td>{j.stiffness}</td>
                      <td>{j.damping}</td>
                      <td>{j.active ? "Muscle" : "Passive"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      )}
    </>
  );
}
