import { Suspense, useEffect, useMemo, useState } from "react";
import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Html } from "@react-three/drei/web/Html";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import * as THREE from "three";
import type { BodyPose } from "../lib/types";
import "./Vision3D.css";

type Geometry = {
  model: string;
  eyes: { origins: number[][]; directions: number[][] }[];
};
const COLORS = ["#74d7ed", "#ffbd70"];
const quaternion = (q: number[]) =>
  new THREE.Quaternion(q[1], q[2], q[3], q[0]);

function Anatomy({ pose, head }: { pose: BodyPose; head: BodyPose }) {
  const mesh = useLoader(STLLoader, `/models/${pose.mesh}.stl`);
  const inverse = quaternion(head.quaternion).invert();
  const position = new THREE.Vector3(...pose.position)
    .sub(new THREE.Vector3(...head.position))
    .applyQuaternion(inverse);
  const rotation = inverse.multiply(quaternion(pose.quaternion));
  const side = pose.name === "l_eye" ? 0 : pose.name === "r_eye" ? 1 : -1;
  return (
    <mesh
      geometry={mesh}
      position={position}
      quaternion={rotation}
      scale={[1000, pose.mirror ? -1000 : 1000, 1000]}
    >
      <meshStandardMaterial
        color={side >= 0 ? COLORS[side] : "#829399"}
        transparent
        opacity={side >= 0 ? 0.95 : 0.36}
        depthWrite={side >= 0}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function EyeRays({
  eye,
  side,
  lines,
}: {
  eye: Geometry["eyes"][number];
  side: number;
  lines: boolean;
}) {
  const { segments, points, origins } = useMemo(() => {
    const starts: number[] = [],
      ends: number[] = [],
      links: number[] = [];
    eye.origins.forEach((origin, i) => {
      const end = origin.map((v, j) => v + eye.directions[i][j] * 1.8);
      starts.push(...origin);
      ends.push(...end);
      links.push(...origin, ...end);
    });
    const make = (a: number[]) =>
      new THREE.BufferGeometry().setAttribute(
        "position",
        new THREE.Float32BufferAttribute(a, 3),
      );
    return { segments: make(links), points: make(ends), origins: make(starts) };
  }, [eye]);
  useEffect(
    () => () => {
      segments.dispose();
      points.dispose();
      origins.dispose();
    },
    [segments, points, origins],
  );
  return (
    <group>
      {lines && (
        <lineSegments geometry={segments}>
          <lineBasicMaterial
            color={COLORS[side]}
            transparent
            opacity={0.1}
            depthWrite={false}
          />
        </lineSegments>
      )}
      <points geometry={points}>
        <pointsMaterial color={COLORS[side]} size={0.018} sizeAttenuation />
      </points>
      <points geometry={origins}>
        <pointsMaterial color={COLORS[side]} size={0.012} sizeAttenuation />
      </points>
    </group>
  );
}

function CameraView({ preset }: { preset: string }) {
  const { camera } = useThree();
  useEffect(() => {
    const positions: Record<string, [number, number, number]> = {
      Oblique: [3.7, 3.2, 2.7],
      Front: [6, 0, 0.1],
      Side: [0, 6, 0.1],
      Rear: [-6, 0, 0.1],
      Top: [0, 0.01, 6],
    };
    camera.up.set(0, 0, 1);
    camera.position.set(...positions[preset]);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, preset]);
  return (
    <OrbitControls
      makeDefault
      target={[0, 0, 0]}
      minDistance={0.5}
      maxDistance={12}
      enablePan={false}
    />
  );
}

export default function Vision3D({
  bodies,
  model,
}: {
  bodies: BodyPose[];
  model: string;
}) {
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [error, setError] = useState("");
  const [eye, setEye] = useState("Both");
  const [preset, setPreset] = useState("Oblique");
  const [lines, setLines] = useState(true);
  useEffect(() => {
    let active = true;
    setGeometry(null);
    setError("");
    fetch(`/models/vision/${encodeURIComponent(model)}.json`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Cannot load optical placement");
        const value = (await response.json()) as Geometry;
        if (value.model !== model) throw new Error("Optical model mismatch");
        return value;
      })
      .then((g) => {
        if (active) setGeometry(g);
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [model]);
  const head = bodies.find((b) => b.name === "c_head");
  return (
    <div className="vision-3d">
      <div className="vision-3d-tools">
        <div role="group" aria-label="Viewing directions shown">
          {["Both", "Left", "Right"].map((v) => (
            <button key={v} aria-pressed={eye === v} onClick={() => setEye(v)}>
              {v}
            </button>
          ))}
        </div>
        <div role="group" aria-label="Eye camera view">
          {["Oblique", "Front", "Side", "Rear", "Top"].map((v) => (
            <button
              key={v}
              aria-pressed={preset === v}
              onClick={() => setPreset(v)}
            >
              {v}
            </button>
          ))}
        </div>
        <label>
          <input
            type="checkbox"
            checked={lines}
            onChange={(e) => setLines(e.target.checked)}
          />
          Direction lines
        </label>
      </div>
      <div
        className="vision-3d-stage"
        role="img"
        aria-label="Interactive 3D fly anatomy with eye positions and viewing directions"
      >
        {error ? (
          <p role="alert">Eye geometry unavailable: {error}</p>
        ) : !geometry || !head ? (
          <p>Loading eye geometry…</p>
        ) : (
          <Canvas
            frameloop="demand"
            camera={{
              position: [3.7, 3.2, 2.7],
              up: [0, 0, 1],
              fov: 48,
              near: 0.01,
              far: 40,
            }}
            dpr={[1, 1.5]}
          >
            <ambientLight intensity={1.4} />
            <directionalLight position={[3, 2, 5]} intensity={2} />
            <Suspense fallback={<Html center>Loading fly meshes…</Html>}>
              {bodies.map((p) => (
                <Anatomy key={p.name} pose={p} head={head} />
              ))}
            </Suspense>
            {geometry.eyes.map(
              (g, i) =>
                (eye === "Both" || eye === (i === 0 ? "Left" : "Right")) && (
                  <EyeRays key={i} eye={g} side={i} lines={lines} />
                ),
            )}
            <Html position={[2.35, 0, 0]} center>
              <span className="vision-direction">Front</span>
            </Html>
            <Html position={[-2.5, 0, 0]} center>
              <span className="vision-direction">Rear</span>
            </Html>
            <Html position={[0, 2.35, 0]} center>
              <span className="vision-direction">Left</span>
            </Html>
            <Html position={[0, -2.35, 0]} center>
              <span className="vision-direction">Right</span>
            </Html>
            <CameraView preset={preset} />
          </Canvas>
        )}
      </div>
      <div className="vision-3d-caption">
        <span style={{ color: COLORS[0] }}>● Left eye</span>
        <span style={{ color: COLORS[1] }}>● Right eye</span>
        <span>Drag to orbit · scroll to zoom</span>
      </div>
      <p className="vision-3d-note">
        {model === "compound-retina-balanced-v4"
          ? "Dots mark unit centers; each unit samples a 3 × 3 patch. Lens positions cover the outward eye surface with a small margin."
          : "Dots mark viewing directions."}{" "}
        Line length is illustrative. Body surfaces can block directions.
        Eye-to-mesh alignment is approximate.
      </p>
    </div>
  );
}
