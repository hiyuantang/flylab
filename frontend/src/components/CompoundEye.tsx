import { useEffect, useMemo, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import * as THREE from "three";
import type { SensoryFrame } from "../lib/types";
import "./CompoundEye.css";

type Surface = {
  surface_vertices: number[][];
  surface_faces: number[][];
  display_positions: number[][];
};

function Retina({ surface, values }: { surface: Surface; values: number[] }) {
  const { shell, pixels } = useMemo(() => {
    const vertices = new Float32Array(surface.surface_vertices.flat());
    const shell = new THREE.BufferGeometry();
    shell.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    shell.setIndex(surface.surface_faces.flat());
    shell.computeVertexNormals();
    shell.computeBoundingBox();
    const box = shell.boundingBox!;
    const center = box.getCenter(new THREE.Vector3());
    const extent = box.getSize(new THREE.Vector3());
    const scale = 2 / Math.max(extent.x, extent.y, extent.z);
    shell.translate(-center.x, -center.y, -center.z);
    shell.scale(scale, scale, scale);
    const positions = surface.display_positions.flatMap((p) =>
      p.map((v, i) => (v - center.getComponent(i)) * scale),
    );
    const pixels = new THREE.BufferGeometry();
    pixels.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    pixels.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(new Float32Array(positions.length), 3),
    );
    return { shell, pixels };
  }, [surface]);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    const color = pixels.getAttribute("color") as THREE.BufferAttribute;
    values.forEach((value, i) => {
      const intensity = Math.max(0, Math.min(1, value));
      color.setXYZ(i, intensity, intensity, intensity);
    });
    color.needsUpdate = true;
    invalidate();
  }, [values, pixels, invalidate]);
  useEffect(
    () => () => {
      shell.dispose();
      pixels.dispose();
    },
    [shell, pixels],
  );
  return (
    <>
      <mesh geometry={shell}>
        <meshStandardMaterial
          color="#18232d"
          roughness={1}
          side={THREE.DoubleSide}
        />
      </mesh>
      <points geometry={pixels} renderOrder={1}>
        <pointsMaterial
          vertexColors
          size={0.016}
          sizeAttenuation
          toneMapped={false}
          depthWrite={false}
        />
      </points>
    </>
  );
}

export function CompoundEye({
  eye,
  side,
  channel,
  model,
}: {
  eye: SensoryFrame["vision"]["eyes"][number];
  side: number;
  channel: string;
  model: string;
}) {
  const [surface, setSurface] = useState<Surface | null>(null);
  const [error, setError] = useState("");
  const [reset, setReset] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    setSurface(null);
    setError("");
    fetch(`/models/vision/${encodeURIComponent(model)}.json`, {
      signal: abort.signal,
    })
      .then(async (r) => {
        if (!r.ok) throw new Error("Eye surface unavailable");
        const data = await r.json();
        const value = data.eyes?.[side] as Surface;
        if (
          data.model !== model ||
          !value?.display_positions ||
          !value.surface_vertices ||
          !value.surface_faces
        )
          throw new Error("Eye surface data does not match this model");
        if (!abort.signal.aborted) setSurface(value);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(String(e));
      });
    return () => abort.abort();
  }, [model, side]);
  const values = eye.patch_channels?.[channel].flat() ?? eye.channels[channel];
  const valid = surface?.display_positions.length === values.length;
  return (
    <figure className="compound-eye curved-eye">
      <figcaption>
        {side === 0 ? "Left" : "Right"} compound eye
        <span>
          {eye.count.toLocaleString()}{" "}
          {eye.patch_channels
            ? "units × 9 pixels · 64 groups × 16"
            : "visual units"}
        </span>
      </figcaption>
      <div
        className="curved-retina"
        role="img"
        aria-label={`${side === 0 ? "Left" : "Right"} eye surface with live grayscale samples; drag to rotate`}
      >
        {error ? (
          <p role="alert">{error}</p>
        ) : !surface ? (
          <p>Loading eye surface…</p>
        ) : !valid ? (
          <p role="alert">Retinal samples do not match this eye surface.</p>
        ) : (
          <Canvas
            key={`${model}-${reset}`}
            frameloop="demand"
            dpr={[1, 1.5]}
            camera={{
              position: [0.35, side === 0 ? 3.3 : -3.3, 0.25],
              up: [0, 0, 1],
              fov: 42,
              near: 0.01,
              far: 20,
            }}
          >
            <ambientLight intensity={0.8} />
            <directionalLight
              position={[2, side === 0 ? 3 : -3, 3]}
              intensity={1.3}
            />
            <Retina surface={surface} values={values} />
            <OrbitControls
              enablePan={false}
              minDistance={1.5}
              maxDistance={6}
            />
          </Canvas>
        )}
      </div>
      <div className="curved-eye-footer">
        <span>Drag to rotate · scroll to zoom</span>
        <button
          onClick={() => setReset((n) => n + 1)}
          aria-label={`Reset ${side === 0 ? "left" : "right"} eye view`}
        >
          Reset view
        </button>
      </div>
    </figure>
  );
}
