import Select from "./Select";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { BufferAttribute, BufferGeometry, Color, Vector3 } from "three";
import { Maximize2, X, Search, Crosshair } from "lucide-react";
import type { NeuralLayout, NeuronConnections } from "../lib/neuralView";

type Scope = "brain" | "vnc" | "all";
const number = (n: number) => n.toLocaleString();
// Rigid axis permutation/reflection and one uniform scale, shared by every point.
// EM x -> screen x, EM z -> -screen y, EM y -> depth. No regional repositioning.
const position = (p: ArrayLike<number>) =>
  new Vector3(
    (p[0] - 48000) / 10000,
    -(p[2] - 40000) / 10000,
    (p[1] - 35000) / 10000,
  );
const palette = [
  new Color("#25465d"),
  new Color("#379bd9"),
  new Color("#aedb8f"),
  new Color("#ffc27b"),
];
async function checked(response: Response) {
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.detail ?? "Could not load measured network.");
  }
  return response;
}

function CameraFit({
  center,
  extent,
  focus,
}: {
  center: Vector3;
  extent: number;
  focus: Vector3 | null;
}) {
  const { camera, controls, invalidate } = useThree();
  useEffect(() => {
    const target = focus ?? center;
    const distance = focus ? 1.4 : extent * 1.6;
    camera.position.set(target.x, target.y, target.z + distance);
    camera.lookAt(target);
    const orbit = controls as unknown as {
      target: Vector3;
      update: () => void;
    } | null;
    orbit?.target.copy(target);
    orbit?.update();
    camera.updateProjectionMatrix();
    invalidate();
  }, [camera, controls, center, extent, focus, invalidate]);
  return null;
}

const NeuronCloud = memo(function NeuronCloud({
  layout,
  rates,
  scope,
  connections,
  onSelect,
  focus,
}: {
  layout: NeuralLayout;
  rates: Float32Array | null;
  scope: Scope;
  connections: NeuronConnections | null;
  onSelect: (id: number) => void;
  focus: Vector3 | null;
}) {
  const { invalidate } = useThree();
  const geometry = useMemo(() => {
    const indices: number[] = [],
      points: number[] = [];
    let min = new Vector3(Infinity, Infinity, Infinity),
      max = new Vector3(-Infinity, -Infinity, -Infinity);
    for (let i = 0; i < layout.neurons; i++) {
      if (
        !Number.isFinite(layout.positions[i * 3]) ||
        (scope === "brain" && layout.vnc[i]) ||
        (scope === "vnc" && !layout.vnc[i])
      )
        continue;
      const p = position(layout.positions.subarray(i * 3, i * 3 + 3));
      indices.push(i);
      points.push(p.x, p.y, p.z);
      min.min(p);
      max.max(p);
    }
    const g = new BufferGeometry();
    g.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(points), 3),
    );
    g.setAttribute(
      "color",
      new BufferAttribute(new Float32Array(points.length), 3),
    );
    g.computeBoundingSphere();
    return {
      g,
      indices,
      center: indices.length
        ? min.clone().add(max).multiplyScalar(0.5)
        : new Vector3(),
      extent: indices.length
        ? Math.max(...max.clone().sub(min).toArray(), 1)
        : 4,
    };
  }, [layout, scope]);
  useEffect(() => () => geometry.g.dispose(), [geometry]);
  useEffect(() => {
    const colors = geometry.g.getAttribute("color") as BufferAttribute;
    const c = new Color();
    for (let j = 0; j < geometry.indices.length; j++) {
      const rate = rates?.[geometry.indices[j]] ?? 0;
      const scale = Math.min(
        3,
        (Math.log1p(Math.max(0, rate)) / Math.log(101)) * 3,
      );
      const lo = Math.min(2, Math.floor(scale));
      c.copy(palette[lo]).lerp(palette[lo + 1], scale - lo);
      colors.setXYZ(j, c.r, c.g, c.b);
    }
    colors.needsUpdate = true;
    invalidate();
  }, [geometry, rates, invalidate]);
  const links = useMemo(() => {
    const lines: number[] = [],
      partners: number[] = [];
    if (connections?.neuron.position) {
      const selected = position(connections.neuron.position);
      for (const p of connections.partners) {
        if (!p.position) continue;
        const neighbor = position(p.position);
        partners.push(...neighbor.toArray());
        const from = connections.direction === "incoming" ? neighbor : selected;
        const to = connections.direction === "incoming" ? selected : neighbor;
        lines.push(...from.toArray(), ...to.toArray());
        const delta = to.clone().sub(from);
        if (delta.length() < 0.001) continue;
        const tip = from.clone().addScaledVector(delta, 0.78);
        const back = delta
          .clone()
          .normalize()
          .multiplyScalar(Math.min(0.045, delta.length() * 0.14));
        let perpendicular = delta.clone().cross(new Vector3(0, 0, 1));
        if (perpendicular.lengthSq() < 1e-8)
          perpendicular = delta.clone().cross(new Vector3(0, 1, 0));
        perpendicular.normalize().multiplyScalar(back.length() * 0.5);
        lines.push(
          ...tip.toArray(),
          ...tip.clone().sub(back).add(perpendicular).toArray(),
          ...tip.toArray(),
          ...tip.clone().sub(back).sub(perpendicular).toArray(),
        );
      }
    }
    return {
      lines: new Float32Array(lines),
      partners: new Float32Array(partners),
    };
  }, [connections]);
  return (
    <>
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.15}
        minDistance={0.08}
        maxDistance={45}
      />
      <CameraFit
        center={geometry.center}
        extent={geometry.extent}
        focus={focus}
      />
      <points
        geometry={geometry.g}
        onClick={(e) => {
          if (e.index !== undefined) {
            e.stopPropagation();
            onSelect(layout.ids[geometry.indices[e.index]]);
          }
        }}
      >
        <pointsMaterial
          vertexColors
          size={1.2}
          sizeAttenuation={false}
          transparent
          opacity={0.85}
          depthWrite={false}
          toneMapped={false}
        />
      </points>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[links.lines, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={connections?.direction === "incoming" ? "#66d9ef" : "#ffc27b"}
          transparent
          opacity={0.5}
          depthWrite={false}
          toneMapped={false}
        />
      </lineSegments>
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[links.partners, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color={connections?.direction === "incoming" ? "#66d9ef" : "#ffc27b"}
          size={5}
          sizeAttenuation={false}
          depthTest={false}
          toneMapped={false}
        />
      </points>
      {connections?.neuron.position && (
        <mesh position={position(connections.neuron.position)} renderOrder={5}>
          <sphereGeometry args={[0.032, 12, 8]} />
          <meshBasicMaterial
            color="#ffffff"
            depthTest={false}
            toneMapped={false}
          />
        </mesh>
      )}
    </>
  );
});

export function NeuralNetwork({
  available,
  stamp,
}: {
  available: boolean;
  stamp: string;
}) {
  const [layout, setLayout] = useState<NeuralLayout | null>(null);
  const [rates, setRates] = useState<Float32Array | null>(null);
  const [time, setTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [scope, setScope] = useState<Scope>("all");
  const [expanded, setExpanded] = useState(false);
  const [selection, setSelection] = useState<{
    id: number;
    direction: "incoming" | "outgoing";
    offset: number;
  } | null>(null);
  const [connections, setConnections] = useState<NeuronConnections | null>(
    null,
  );
  const [detailError, setDetailError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [focus, setFocus] = useState<Vector3 | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const stampRef = useRef(stamp);
  stampRef.current = stamp;
  const [detailRefresh, setDetailRefresh] = useState(0);
  useEffect(() => {
    if (!available) return;
    const abort = new AbortController();
    setError(null);
    setLayout(null);
    setRates(null);
    setTime(null);
    Promise.all([
      fetch("/api/neural-view", { signal: abort.signal })
        .then(checked)
        .then((r) => r.json()),
      fetch("/api/neural-view/layout", { signal: abort.signal })
        .then(checked)
        .then(async (r) => ({
          buffer: await r.arrayBuffer(),
          hash: r.headers.get("X-Graph-SHA256"),
        })),
    ])
      .then(([meta, { buffer, hash }]) => {
        if (
          meta.graph_sha256 !== hash ||
          buffer.byteLength !== meta.neurons * 21
        )
          throw new Error("Neuron layout version mismatch. Retry loading.");
        if (!abort.signal.aborted)
          setLayout({
            ...meta,
            ids: new Float64Array(buffer, 0, meta.neurons),
            positions: new Float32Array(
              buffer,
              meta.neurons * 8,
              meta.neurons * 3,
            ),
            vnc: new Uint8Array(buffer, meta.neurons * 20, meta.neurons),
          });
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(e.message);
      });
    return () => abort.abort();
  }, [available, retry]);
  useEffect(() => {
    if (!layout || !available) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>,
      previous = "";
    const poll = async () => {
      try {
        if (!document.hidden && previous !== stampRef.current) {
          const requestedStamp = stampRef.current;
          const response = await checked(
            await fetch("/api/neural-view/activity", { signal: abort.signal }),
          );
          if (response.headers.get("X-Graph-SHA256") !== layout.graph_sha256)
            throw new Error(
              "The loaded graph changed. Retry loading the view.",
            );
          const buffer = await response.arrayBuffer();
          if (buffer.byteLength !== layout.neurons * 4)
            throw new Error("Incomplete neural activity snapshot.");
          if (!abort.signal.aborted) {
            setRates(new Float32Array(buffer));
            setTime(Number(response.headers.get("X-Neural-Time-Ms")));
            setError(null);
            setDetailRefresh((x) => x + 1);
            previous = requestedStamp;
          }
        }
      } catch (e) {
        if (!abort.signal.aborted) setError((e as Error).message);
      } finally {
        if (!abort.signal.aborted) timer = setTimeout(poll, 1000);
      }
    };
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [available, layout]);
  useEffect(() => {
    if (!selection || !available || !layout) return;
    const abort = new AbortController();
    setDetailError(null);
    fetch(
      `/api/neural-view/neuron/${selection.id}?direction=${selection.direction}&offset=${selection.offset}&limit=100`,
      { signal: abort.signal },
    )
      .then(checked)
      .then((r) => r.json())
      .then((data: NeuronConnections) => {
        if (data.graph_sha256 !== layout.graph_sha256)
          throw new Error(
            "Connection graph differs from the displayed layout.",
          );
        if (!abort.signal.aborted) setConnections(data);
      })
      .catch((e) => {
        if (!abort.signal.aborted) {
          setConnections(null);
          setDetailError(e.message);
        }
      });
    return () => abort.abort();
  }, [selection, available, layout, detailRefresh]);
  useEffect(() => {
    if (expanded) dialog.current?.showModal();
  }, [expanded]);
  const select = useCallback((id: number) => {
    setSelection({ id, direction: "incoming", offset: 0 });
    setConnections(null);
    setQuery(String(id));
    setFocus(null);
  }, []);
  const visibleCount = useMemo(
    () =>
      layout
        ? Array.from(layout.vnc).reduce(
            (sum, v, i) =>
              sum +
              (Number.isFinite(layout.positions[i * 3]) &&
              (scope === "all" || (scope === "vnc" ? !!v : !v))
                ? 1
                : 0),
            0,
          )
        : 0,
    [layout, scope],
  );
  const content = (
    <div className={`neural-network ${expanded ? "expanded" : ""}`}>
      <div className="neural-tools">
        <label>
          Show{" "}
          <Select
            aria-label="Neural spatial scope"
            value={scope}
            onChange={(value) => {
              setScope(value as Scope);
              setFocus(null);
            }}
            options={[
              { value: "all", label: "Whole CNS · brain + VNC" },
              { value: "brain", label: "Brain classes" },
              { value: "vnc", label: "VNC classes" },
            ]}
          />
        </label>
        <span className="mono">
          {time === null
            ? "Awaiting state"
            : `${(time / 1000).toFixed(2)} s · live state`}
        </span>
        <button
          aria-label={expanded ? "Close enlarged brain" : "Enlarge brain"}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <X size={16} /> : <Maximize2 size={16} />}
        </button>
      </div>
      {!available ? (
        <p className="neural-message">
          Connect to the measured brain to view its neurons.
        </p>
      ) : error ? (
        <p role="alert" className="neural-message">
          {error}{" "}
          <button onClick={() => setRetry((x) => x + 1)}>Retry view</button>
        </p>
      ) : !layout ? (
        <p role="status" className="neural-message">
          Loading measured soma locations…
        </p>
      ) : (
        <>
          <div className="neural-workspace">
            <div className="neural-stage">
              <Canvas
                frameloop="demand"
                dpr={[1, 1.5]}
                camera={{ position: [0, 0, 14], fov: 45, near: 0.01, far: 150 }}
                raycaster={{
                  params: {
                    Mesh: {},
                    LOD: {},
                    Sprite: {},
                    Points: { threshold: 0.025 },
                    Line: { threshold: 0.01 },
                  },
                }}
              >
                <NeuronCloud
                  layout={layout}
                  rates={rates}
                  scope={scope}
                  connections={connections}
                  onSelect={select}
                  focus={focus}
                />
              </Canvas>
              <div className="neural-stage-label">
                {number(visibleCount)} measured soma points
                <br />
                <span>Drag to orbit · scroll to zoom · click a neuron</span>
              </div>
              <div className="neural-color-key">
                <span>0</span>
                <i />
                <span>100+ Hz</span>
              </div>
            </div>
            <aside
              className="neural-inspector"
              aria-label="Neuron connection inspector"
            >
              <form
                className="anatomy-search"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (
                    /^\d+$/.test(query.trim()) &&
                    Number.isSafeInteger(Number(query))
                  )
                    select(Number(query));
                  else setDetailError("Enter a numeric MaleCNS neuron ID.");
                }}
              >
                <input
                  aria-label="Find neuron by ID"
                  placeholder="Find neuron by MaleCNS ID"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button aria-label="Find neuron">
                  <Search size={15} />
                </button>
              </form>
              {detailError && <p role="alert">{detailError}</p>}
              {!selection ? (
                <div className="neural-empty">
                  <strong>Follow a neuron’s connections</strong>
                  <p>
                    Select a point or enter an ID. Inspect incoming and outgoing
                    partners from the exact simulated graph.
                  </p>
                  <p>
                    Points mark cell bodies. Connection lines are graph links,
                    not reconstructed axons.
                  </p>
                </div>
              ) : !connections ? (
                !detailError && <p role="status">Loading neuron connections…</p>
              ) : (
                <>
                  <div className="neural-selected">
                    <div>
                      <strong>
                        {connections.neuron.type ?? "Unidentified type"}
                      </strong>
                      <span className="mono">
                        #{connections.neuron.body_id}
                      </span>
                    </div>
                    <button
                      aria-label="Focus selected neuron"
                      disabled={!connections.neuron.position}
                      onClick={() =>
                        setFocus(position(connections.neuron.position!))
                      }
                    >
                      <Crosshair size={16} />
                    </button>
                  </div>
                  <p className="neural-identity">
                    {connections.neuron.superclass} ·{" "}
                    {connections.neuron.transmitter ?? "Unknown transmitter"} ·{" "}
                    {connections.neuron.soma_side ?? "?"} soma
                  </p>
                  <div className="neural-readout">
                    <span>
                      <b>
                        {(
                          rates?.[connections.neuron.index] ??
                          connections.neuron.rate_hz
                        ).toFixed(2)}
                      </b>{" "}
                      Hz
                    </span>
                    <span>
                      <b>{connections.neuron.voltage.toFixed(2)}</b>{" "}
                      {connections.neuron.voltage_unit}
                    </span>
                    <span>
                      <b>{number(connections.neuron.spike_count)}</b> spikes
                    </span>
                  </div>
                  <p className="hint">
                    Detail sampled at{" "}
                    {(connections.neural_time_ms / 1000).toFixed(2)} s.{" "}
                    {!connections.neuron.position &&
                      "No measured soma location; connections remain inspectable."}
                  </p>
                  <div
                    className="brain-view-tabs"
                    role="group"
                    aria-label="Connection direction"
                  >
                    {(["incoming", "outgoing"] as const).map((direction) => (
                      <button
                        key={direction}
                        aria-pressed={selection.direction === direction}
                        onClick={() => {
                          setConnections(null);
                          setSelection({ ...selection, direction, offset: 0 });
                        }}
                      >
                        {direction === "incoming" ? "Incoming" : "Outgoing"} ·{" "}
                        {number(
                          direction === "incoming"
                            ? connections.incoming_total
                            : connections.outgoing_total,
                        )}
                      </button>
                    ))}
                  </div>
                  <p className="hint">
                    {number(
                      selection.direction === "incoming"
                        ? connections.incoming_synapses
                        : connections.outgoing_synapses,
                    )}{" "}
                    synapses · sorted by synapse count
                  </p>
                  <div
                    className="neural-partners"
                    aria-label="Connected neurons"
                  >
                    {connections.partners.length ? (
                      connections.partners.map((p) => (
                        <button
                          key={p.body_id}
                          onClick={() => select(p.body_id)}
                        >
                          <span>
                            <strong>{p.type ?? "Unidentified"}</strong>
                            <small>
                              #{p.body_id}
                              {!p.position ? " · no location" : ""}
                            </small>
                          </span>
                          <span className="mono">
                            {number(p.synapses)}
                            <small>synapses</small>
                          </span>
                        </button>
                      ))
                    ) : (
                      <p>
                        No {selection.direction} partners in the imported graph.
                      </p>
                    )}
                  </div>
                  <div className="neural-pagination">
                    <button
                      disabled={selection.offset === 0}
                      onClick={() => {
                        setConnections(null);
                        setSelection({
                          ...selection,
                          offset: Math.max(0, selection.offset - 100),
                        });
                      }}
                    >
                      Previous
                    </button>
                    <span>
                      {connections.total ? selection.offset + 1 : 0}–
                      {Math.min(selection.offset + 100, connections.total)} /{" "}
                      {number(connections.total)}
                    </span>
                    <button
                      disabled={selection.offset + 100 >= connections.total}
                      onClick={() => {
                        setConnections(null);
                        setSelection({
                          ...selection,
                          offset: selection.offset + 100,
                        });
                      }}
                    >
                      Next
                    </button>
                  </div>
                  <p className="hint">
                    Only this page’s located partners are drawn. Arrows point
                    from sender to receiver. All partners remain available
                    through pagination.
                  </p>
                </>
              )}
            </aside>
          </div>
          <p className="neural-provenance">
            {number(layout.positioned)} / {number(layout.neurons)} neurons have
            measured soma locations; {number(layout.missing_positions)} remain
            searchable without invented positions.{" "}
            <a href={layout.source} target="_blank" rel="noreferrer">
              MaleCNS v1.0
            </a>{" "}
            · 8 nm voxel space. Brain/VNC filters use annotated classes, not
            neuropil boundaries. Colors show simulated firing rates; display
            refresh is capped at 1 Hz.
          </p>
        </>
      )}
    </div>
  );
  return expanded ? (
    <>
      <div className="neural-message">Brain open in enlarged view.</div>
      {createPortal(
        <dialog
          ref={dialog}
          className="neural-dialog"
          aria-label="Enlarged measured brain"
          onCancel={() => setExpanded(false)}
          onClose={() => setExpanded(false)}
        >
          {content}
        </dialog>,
        document.body,
      )}
    </>
  ) : (
    content
  );
}
