import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Zap, VolumeX } from "lucide-react";
import type { Simulation, Command, Region, RegionId } from "../lib/types";
const geometries: Record<string, { p: number[]; s: number[] }[]> = {
  optic: [
    { p: [-1.8, 0, 0], s: [0.7, 0.87, 0.5] },
    { p: [1.8, 0, 0], s: [0.7, 0.87, 0.5] },
  ],
  antennal: [
    { p: [-0.45, -0.65, 0.45], s: [0.4, 0.4, 0.34] },
    { p: [0.45, -0.65, 0.45], s: [0.4, 0.4, 0.34] },
  ],
  mushroom: [
    { p: [-0.65, 0.52, 0.38], s: [0.3, 0.5, 0.28] },
    { p: [0.65, 0.52, 0.38], s: [0.3, 0.5, 0.28] },
  ],
  descending: [
    { p: [-0.27, -0.82, 0], s: [0.2, 0.55, 0.2] },
    { p: [0.27, -0.82, 0], s: [0.2, 0.55, 0.2] },
  ],
  vnc: [{ p: [0, -1.65, 0], s: [0.4, 0.64, 0.3] }],
  motor: [
    { p: [-0.35, -1.85, 0.18], s: [0.17, 0.5, 0.2] },
    { p: [0.35, -1.85, 0.18], s: [0.17, 0.5, 0.2] },
  ],
};
function Cloud({
  region,
  selected,
  onSelect,
}: {
  region: Region;
  selected: boolean;
  onSelect: () => void;
}) {
  const points = useMemo(() => {
    let seed = region.start + 10;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const values: number[] = [];
    for (const shape of geometries[region.id])
      for (let i = 0; i < 650; i++) {
        const theta = rand() * Math.PI * 2;
        const z = rand() * 2 - 1;
        const r = Math.pow(rand(), 0.25);
        const radial = Math.sqrt(1 - z * z);
        values.push(
          shape.p[0] + Math.cos(theta) * radial * shape.s[0] * r,
          shape.p[1] + z * shape.s[1] * r,
          shape.p[2] + Math.sin(theta) * radial * shape.s[2] * r,
        );
      }
    return new Float32Array(values);
  }, [region.id, region.start]);

  return (
    <group>
      <points
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
      >
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[points, 3]} />
        </bufferGeometry>
        <pointsMaterial
          transparent
          depthWrite={false}
          toneMapped={false}
          sizeAttenuation
          size={selected ? 0.045 : 0.03}
          opacity={0.7 + region.activity * 0.3}
          color={selected || region.stimulated ? "#ff9f67" : "#328fff"}
        />
      </points>
      {geometries[region.id].map((shape, i) => (
        <mesh
          key={i}
          position={shape.p as [number, number, number]}
          scale={shape.s as [number, number, number]}
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
        >
          <sphereGeometry args={[1, 16, 16]} />
          <meshBasicMaterial
            color="#4a94dd"
            wireframe
            transparent
            opacity={selected ? 0.12 : 0.035}
          />
        </mesh>
      ))}
    </group>
  );
}
function BrainEnvelope() {
  const points = useMemo(() => {
    const values: number[] = [];
    for (const side of [-1, 1])
      for (let i = 0; i < 1600; i++) {
        const z = 1 - (2 * (i + 0.5)) / 1600;
        const angle = i * 2.399963;
        const radius = Math.sqrt(1 - z * z);
        values.push(
          side * 0.72 + Math.cos(angle) * radius * 0.95,
          z * 0.94 + 0.12,
          Math.sin(angle) * radius * 0.4 - 0.12,
        );
      }
    return new Float32Array(values);
  }, []);
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[points, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#368ee9"
        toneMapped={false}
        size={0.021}
        opacity={0.3}
        transparent
        depthWrite={false}
      />
    </points>
  );
}
export function Brain({
  simulation,
  command,
  disabled,
}: {
  simulation: Simulation;
  command: Command;
  disabled: boolean;
}) {
  const [amplitude, setAmplitude] = useState(1);
  const selected = simulation.regions.find(
    (r) => r.id === simulation.selected,
  )!;
  return (
    <section className="panel brain-panel">
      <div className="panel-heading">
        <div>
          <h2>Brain activity</h2>
          <p>Schematic regions · live neural state</p>
        </div>
        <span className="live-dot" title="Live neural state" />
      </div>
      <div className="brain-canvas">
        <Canvas camera={{ position: [0, -0.25, 6.2], fov: 40 }} dpr={[1, 1.5]}>
          <BrainEnvelope />
          {simulation.regions.map((r) => (
            <Cloud
              key={r.id}
              region={r}
              selected={r.id === simulation.selected}
              onSelect={() => {
                if (!disabled) void command("select", { region: r.id });
              }}
            />
          ))}
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            target={[0, -0.4, 0]}
            minPolarAngle={Math.PI * 0.3}
            maxPolarAngle={Math.PI * 0.7}
          />
        </Canvas>
        <span className="hemisphere left">L</span>
        <span className="hemisphere right">R</span>
        <div className="brain-legend">
          <i /> {selected.name}
        </div>
      </div>
      <div className="region-list">
        {simulation.regions.map((r) => (
          <button
            key={r.id}
            disabled={disabled}
            onClick={() => command("select", { region: r.id })}
            className={`region-row ${r.id === simulation.selected ? "selected" : ""}`}
          >
            <span>
              {r.name}
              {simulation.silenced.includes(r.id) && <VolumeX size={12} />}
            </span>
            <span className="bar-track">
              <span style={{ width: `${Math.min(100, r.activity * 100)}%` }} />
            </span>
            <span className="mono">{r.activity.toFixed(2)}</span>
          </button>
        ))}
      </div>
      <div className="stimulation">
        <div className="field-line">
          <label htmlFor="amplitude">Stimulation strength</label>
          <span className="mono">{amplitude.toFixed(1)}</span>
        </div>
        <input
          id="amplitude"
          type="range"
          min="0.1"
          max="3"
          step="0.1"
          value={amplitude}
          onChange={(e) => setAmplitude(+e.target.value)}
          disabled={disabled}
        />
        <div className="stim-controls">
          <select
            aria-label="Stimulation region"
            value={simulation.selected}
            onChange={(e) =>
              command("select", { region: e.target.value as RegionId })
            }
            disabled={disabled}
          >
            {simulation.regions.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
          <button
            className="primary"
            disabled={disabled}
            onClick={() =>
              command("stimulate", {
                region: simulation.selected,
                amplitude,
                duration: 0.5,
              })
            }
          >
            <Zap size={16} /> Stimulate
          </button>
          <button
            className={
              simulation.silenced.includes(simulation.selected)
                ? "silenced"
                : ""
            }
            title="Toggle region silencing"
            aria-label="Silence selected region"
            aria-pressed={simulation.silenced.includes(simulation.selected)}
            disabled={disabled}
            onClick={() =>
              command("silence", {
                region: simulation.selected,
                enabled: !simulation.silenced.includes(simulation.selected),
              })
            }
          >
            <VolumeX size={17} />
          </button>
        </div>
        <p className="hint">500 ms pulse · advances when the simulation runs</p>
      </div>
    </section>
  );
}
