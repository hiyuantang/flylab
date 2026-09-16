import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Html } from "@react-three/drei/web/Html";
import { BufferGeometry, Float32BufferAttribute, Color } from "three";
import { Maximize2, X, Search, RotateCcw, Network } from "lucide-react";
import Select from "./Select";
import type { PolicyActivity } from "../lib/types";
import "./PolicyNetwork.css";

type Group = PolicyActivity["groups"][number];
type Selection = { group: string; index: number };
const cool = new Color("#2c526a"),
  warm = new Color("#d8ef97");

function Layer({
  group,
  groups,
  values,
  selected,
  onSelect,
}: {
  group: Group;
  groups: Group[];
  values?: number[];
  selected: Selection | null;
  onSelect: (selection: Selection) => void;
}) {
  const peers = groups.filter((g) => g.stage === group.stage);
  const slot = peers.findIndex((g) => g.id === group.id);
  const x = (slot - (peers.length - 1) / 2) * 7.4;
  const y = (Math.max(...groups.map((g) => g.stage)) / 2 - group.stage) * 1.8;
  const columns = peers.length > 1 ? 16 : 32;
  const geometry = useMemo(() => {
    const points: number[] = [],
      colors: number[] = [];
    const maximum =
      group.reduction === "excitation" ? 1 : Math.max(...(values ?? []), 1e-12);
    for (let i = 0; i < group.count; i++) {
      points.push(
        ((i % columns) - (columns - 1) / 2) * 0.37,
        (Math.floor(i / columns) -
          Math.floor((group.count - 1) / columns) / 2) *
          0.16,
        0,
      );
      const chosen = selected?.group === group.id && selected.index === i;
      const c = chosen
        ? new Color("#ffffff")
        : cool
            .clone()
            .lerp(warm, values ? Math.min(1, values[i] / maximum) : 0);
      colors.push(c.r, c.g, c.b);
    }
    const result = new BufferGeometry();
    result.setAttribute("position", new Float32BufferAttribute(points, 3));
    result.setAttribute("color", new Float32BufferAttribute(colors, 3));
    return result;
  }, [group, columns, values, selected]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <group position={[x, y, 0]}>
      <points
        geometry={geometry}
        onClick={(event) => {
          event.stopPropagation();
          if (event.index !== undefined)
            onSelect({ group: group.id, index: event.index });
        }}
      >
        <pointsMaterial size={0.3} vertexColors sizeAttenuation />
      </points>
      <Html
        position={[peers.length > 1 ? 0 : -9, peers.length > 1 ? 0.9 : 0, 0]}
        center
        style={{ pointerEvents: "none" }}
      >
        <div className="policy-layer-label">
          <strong>{group.label}</strong>
        </div>
      </Html>
    </group>
  );
}

function Wiring({ groups }: { groups: Group[] }) {
  const geometry = useMemo(() => {
    const last = Math.max(...groups.map((g) => g.stage));
    const points: number[] = [];
    const location = (g: Group): number[] => {
      const peers = groups.filter((p) => p.stage === g.stage);
      return [
        (peers.indexOf(g) - (peers.length - 1) / 2) * 7.4,
        (last / 2 - g.stage) * 1.8,
        -0.25,
      ];
    };
    groups.forEach((g) =>
      groups
        .filter((next) => next.stage === g.stage + 1)
        .forEach((next) => points.push(...location(g), ...location(next))),
    );
    // Encoder memory is also consumed by every decoder layer.
    const encoder = groups.filter((g) => g.id.startsWith("encoder_")).at(-1);
    if (encoder)
      groups
        .filter((g) => g.id.startsWith("decoder_"))
        .forEach((g) => {
          const a = location(encoder),
            b = location(g);
          points.push(
            ...a,
            7,
            a[1],
            -0.25,
            7,
            a[1],
            -0.25,
            7,
            b[1],
            -0.25,
            7,
            b[1],
            -0.25,
            ...b,
          );
        });
    return new BufferGeometry().setAttribute(
      "position",
      new Float32BufferAttribute(points, 3),
    );
  }, [groups]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#486d82" transparent opacity={0.6} />
    </lineSegments>
  );
}

export default function PolicyNetwork({
  activity,
  parameters,
  modelKey,
}: {
  activity?: PolicyActivity;
  parameters?: number;
  modelKey: string;
}) {
  const [scope, setScope] = useState("all");
  const [selected, setSelected] = useState<Selection | null>(null);
  const [query, setQuery] = useState("");
  const [message, setMessage] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [cameraKey, setCameraKey] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  const expandButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    setSelected(null);
    setScope("all");
    setQuery("");
    setMessage("");
  }, [modelKey]);
  useEffect(() => {
    if (expanded) dialog.current?.showModal();
  }, [expanded]);
  const groups = activity?.groups ?? [];
  const visible = groups.filter((g) => scope === "all" || g.id === scope);
  const group = groups.find((g) => g.id === selected?.group);
  const value = selected
    ? activity?.values[selected.group]?.[selected.index]
    : undefined;
  function close() {
    setExpanded(false);
    requestAnimationFrame(() => expandButton.current?.focus());
  }
  function search() {
    const q = query.trim().toLowerCase();
    if (!q) {
      setSelected(null);
      setMessage("");
      return;
    }
    for (const g of groups) {
      for (let i = 0; i < g.count; i++) {
        const id = `${g.id}:${i + 1}`;
        if (id === q || g.labels?.[i].toLowerCase() === q) {
          setSelected({ group: g.id, index: i });
          setScope(g.id);
          setMessage("");
          return;
        }
      }
    }
    setMessage("Use a point ID, such as encoder_1:1, or an exact muscle name.");
  }
  const content = (
    <section className="panel policy-network">
      <div className="policy-network-heading">
        <div>
          <h2>
            <Network size={19} /> Policy activity
          </h2>
          <p>{parameters?.toLocaleString()} parameters · engineered layout</p>
        </div>
        <button
          ref={expanded ? undefined : expandButton}
          className="icon-button"
          aria-label={
            expanded ? "Close enlarged policy" : "Enlarge policy network"
          }
          onClick={() => (expanded ? close() : setExpanded(true))}
        >
          {expanded ? <X size={18} /> : <Maximize2 size={18} />}
        </button>
      </div>
      <div className="policy-network-tools">
        <Select
          aria-label="Policy section"
          value={scope}
          onChange={setScope}
          options={[
            { value: "all", label: "Whole policy" },
            ...groups.map((g) => ({ value: g.id, label: g.label })),
          ]}
        />
        <form
          onSubmit={(e) => {
            e.preventDefault();
            search();
          }}
        >
          <input
            aria-label="Find policy unit"
            placeholder="Find unit or muscle"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button className="icon-button" aria-label="Search policy units">
            <Search size={16} />
          </button>
        </form>
        <button
          className="icon-button"
          aria-label="Reset policy camera"
          onClick={() => setCameraKey((k) => k + 1)}
        >
          <RotateCcw size={16} />
        </button>
      </div>
      <div className="policy-network-stage">
        {!!groups.length && (
          <Canvas
            key={`${cameraKey}-${scope}`}
            camera={{
              position: scope === "all" ? [2, 0, 24] : [1, 0.5, 15],
              fov: 42,
            }}
            dpr={[1, 1.5]}
            frameloop="demand"
            raycaster={{
              params: {
                Mesh: {},
                LOD: {},
                Sprite: {},
                Points: { threshold: 0.16 },
                Line: { threshold: 0.16 },
              },
            }}
            onPointerMissed={() => {
              setSelected(null);
              setMessage("");
            }}
          >
            <color attach="background" args={["#0b1a25"]} />
            {scope === "all" && <Wiring groups={groups} />}
            {visible.map((g) => (
              <Layer
                key={g.id}
                group={scope === "all" ? g : { ...g, stage: 0 }}
                groups={scope === "all" ? groups : [{ ...g, stage: 0 }]}
                values={activity?.values[g.id]}
                selected={selected}
                onSelect={setSelected}
              />
            ))}
            <OrbitControls
              makeDefault
              enableDamping={false}
              minDistance={4}
              maxDistance={45}
            />
          </Canvas>
        )}
        <div className="policy-network-state">
          {activity?.measured
            ? `Measured forward pass · step ${activity.step}`
            : "Run or step to measure activations"}
        </div>
        <div className="policy-network-legend">
          <i /> Features: relative RMS · commands: 0–1
        </div>
      </div>
      <div className="policy-unit-inspector" aria-live="polite">
        {selected && group ? (
          <>
            <strong>
              {group.labels?.[selected.index] ??
                `${group.label} · ${group.reduction === "token_rms" ? "token" : "feature"} ${selected.index + 1}`}
            </strong>
            <span>
              {group.id}:{selected.index + 1} ·{" "}
              {value === undefined ? "Not measured" : value.toPrecision(6)} ·{" "}
              {group.unit}
            </span>
          </>
        ) : (
          <>
            <strong>
              {message || "Select a point to inspect its activation"}
            </strong>
            <span>
              Drag to orbit · scroll to zoom · click blank space to clear
            </span>
          </>
        )}
      </div>
      <p className="policy-network-note">
        Dots show tokens and feature channels, not biological neurons. RMS
        summarizes activity across each token or channel. Lines show signal
        flow, not attention weights. Muscle dots show predicted excitation, not
        achieved activation.
      </p>
    </section>
  );
  return expanded ? (
    <>
      <div className="panel policy-network-placeholder">
        Policy network open in enlarged view.
      </div>
      {createPortal(
        <dialog
          ref={dialog}
          className="policy-network-dialog"
          aria-label="Enlarged policy network"
          onCancel={close}
          onClose={close}
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
