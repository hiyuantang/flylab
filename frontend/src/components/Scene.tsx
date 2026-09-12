import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Grid } from "@react-three/drei/core/Grid";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { Focus, Layers3, Move, RotateCcw, GitBranch } from "lucide-react";
import type { Simulation, BodyPose } from "../lib/types";

const LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"];
const WARM = new THREE.Color("#c4924d");
const HOT = new THREE.Color("#ff7736");
function partLabel(name: string) {
  const [prefix, link] = name.split("_");
  return `${prefix.length === 2 ? prefix.toUpperCase() + " " : ""}${link === "trochanterfemur" ? "trochanter / femur" : link}`;
}
function materialColor(name: string) {
  if (name.endsWith("eye")) return "#ad2918";
  if (name.endsWith("wing")) return "#cad7db";
  if (name.endsWith("arista")) return "#37281b";
  if (name.startsWith("c_abdomen"))
    return name.endsWith("6") ? "#553a22" : "#946531";
  if (name.includes("tarsus")) return "#bc8e4d";
  return name.includes("tibia") ? "#b38242" : "#a07136";
}
function Part({
  pose,
  activity,
  overlay,
  selected,
  onSelect,
}: {
  pose: BodyPose;
  activity: number;
  overlay: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const source = useLoader(STLLoader, `/models/${pose.mesh}.stl`);
  const material = useRef<THREE.MeshStandardMaterial>(null);
  const geometry = useMemo(() => {
    // Preserve the joint-relative mesh origin and uniform upstream metre→mm scale.
    // MuJoCo and Three.js apply exactly the same body transform and right-side reflection.
    const copy = source.clone();
    copy.deleteAttribute("normal");
    const g = mergeVertices(copy, 1e-8);
    copy.dispose();
    g.computeVertexNormals();
    return g;
  }, [source]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const hairGeometry = useMemo(() => {
    if (!["c_thorax", "c_head"].includes(pose.name)) return null;
    const positions = geometry.getAttribute("position");
    const normals = geometry.getAttribute("normal");
    const lines: number[] = [];
    for (let i = 0; i < positions.count; i += 17) {
      const x = positions.getX(i) * 1000,
        y = positions.getY(i) * 1000,
        z = positions.getZ(i) * 1000;
      if (normals.getZ(i) < 0.3) continue;
      const length = 0.045 + 0.035 * (((i * 13) % 31) / 31);
      lines.push(
        x,
        y,
        z,
        x + normals.getX(i) * length,
        y + normals.getY(i) * length,
        z + normals.getZ(i) * length,
      );
    }
    return new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(lines, 3),
    );
  }, [geometry, pose.name]);
  useEffect(() => () => hairGeometry?.dispose(), [hairGeometry]);
  useFrame(() => {
    if (material.current) {
      material.current.color.set(materialColor(pose.name));
      if (overlay && LEGS.includes(pose.name.slice(0, 2).toUpperCase())) {
        material.current.color.copy(WARM).lerp(HOT, Math.min(1, activity * 3));
      }
    }
  });
  const wing = pose.name.endsWith("wing");
  return (
    <group
      position={pose.position}
      quaternion={[
        pose.quaternion[1],
        pose.quaternion[2],
        pose.quaternion[3],
        pose.quaternion[0],
      ]}
    >
      <mesh
        geometry={geometry}
        scale={[1000, pose.mirror ? -1000 : 1000, 1000]}
        castShadow={!wing}
        receiveShadow
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
        }}
      >
        <meshStandardMaterial
          ref={material}
          color={materialColor(pose.name)}
          roughness={wing ? 0.23 : 0.64}
          metalness={0}
          transparent={wing}
          opacity={wing ? 0.36 : 1}
          depthWrite={!wing}
          side={THREE.DoubleSide}
          emissive={selected ? "#a5c36a" : "#000000"}
          emissiveIntensity={selected ? 0.13 : 0}
        />
      </mesh>
      {hairGeometry && (
        <lineSegments geometry={hairGeometry}>
          <lineBasicMaterial color="#3a2a1b" transparent opacity={0.85} />
        </lineSegments>
      )}
    </group>
  );
}
function Skeleton({ bodies }: { bodies: BodyPose[] }) {
  const geometry = useMemo(() => {
    const lookup = new Map(bodies.map((body) => [body.name, body]));
    const points: number[] = [];
    bodies.forEach((body) => {
      const parent = body.parent ? lookup.get(body.parent) : undefined;
      if (parent) points.push(...parent.position, ...body.position);
    });
    return new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(points, 3),
    );
  }, [bodies]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <>
      <lineSegments geometry={geometry} renderOrder={3}>
        <lineBasicMaterial
          color="#42e3e8"
          depthTest={false}
          transparent
          opacity={0.8}
        />
      </lineSegments>
      {bodies.map((body) => (
        <mesh key={body.name} position={body.position} renderOrder={4}>
          <sphereGeometry args={[0.025, 8, 6]} />
          <meshBasicMaterial color="#bcffff" depthTest={false} />
        </mesh>
      ))}
    </>
  );
}
function CameraRig({
  view,
  follow,
  position,
  heading,
  resetKey,
}: {
  view: string;
  follow: boolean;
  position: number[];
  heading: number;
  resetKey: number;
}) {
  const { camera } = useThree();
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const previous = useRef(new THREE.Vector3());
  const latest = useRef(position);
  // Keep the body and the opened wings centered as the fly turns.
  latest.current = [
    position[0] - 0.8 * Math.cos(heading),
    position[1] - 0.8 * Math.sin(heading),
    position[2],
  ];
  useEffect(() => {
    const p = latest.current;
    const target = new THREE.Vector3(p[0], p[1], 0.7);
    const offsets: Record<string, [number, number, number]> = {
      fly: [5.8, -7.2, 4.5],
      top: [0, -0.001, 10.5],
      side: [0, -9, 0.6],
    };
    camera.position.copy(target).add(new THREE.Vector3(...offsets[view]));
    camera.up.set(0, 0, 1);
    controls.current?.target.copy(target);
    camera.lookAt(target);
    previous.current.set(p[0], p[1], 0);
    controls.current?.update();
  }, [view, camera, resetKey]);
  useFrame(() => {
    const p = latest.current;
    if (follow) {
      const dx = p[0] - previous.current.x,
        dy = p[1] - previous.current.y;
      camera.position.x += dx;
      camera.position.y += dy;
      if (controls.current) {
        controls.current.target.x += dx;
        controls.current.target.y += dy;
      }
    }
    previous.current.set(p[0], p[1], 0);
  });
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      maxPolarAngle={Math.PI * 0.49}
      minDistance={2.4}
      maxDistance={50}
    />
  );
}
function Lighting({ position }: { position: number[] }) {
  const target = useMemo(() => new THREE.Object3D(), []);
  target.position.set(position[0] - 0.5, position[1], 0.6);
  return (
    <>
      <ambientLight intensity={1.1} />
      <hemisphereLight args={["#eaf4fc", "#7b6650", 1.5]} />
      <primitive object={target} />
      <directionalLight
        target={target}
        position={[position[0] + 2, position[1] - 4, 7]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-4}
        shadow-camera-right={4}
        shadow-camera-top={4}
        shadow-camera-bottom={-4}
        shadow-normalBias={0.018}
        shadow-bias={-0.00002}
      />
    </>
  );
}
export function FlyScene({ simulation }: { simulation: Simulation }) {
  const [view, setView] = useState("fly");
  const [muscles, setMuscles] = useState(false);
  const [skeleton, setSkeleton] = useState(false);
  const [follow, setFollow] = useState(true);
  const [selected, setSelected] = useState("c_thorax");
  const [resetKey, setResetKey] = useState(0);
  const selectedPart = simulation.body.bodies.find(
    (body) => body.name === selected,
  );
  const anatomy = simulation.body.anatomy;
  return (
    <section className="panel arena-panel">
      <div className="panel-heading arena-heading">
        <div>
          <h2>Embodied simulation</h2>
          <p>
            {anatomy.segments} connected segments · {anatomy.joints} joint
            degrees of freedom
          </p>
        </div>
        <span
          className="model-tag"
          title="NeuroMechFly research anatomy, derived from a female specimen"
        >
          Research anatomy
        </span>
      </div>
      <div className="scene-wrap">
        <Canvas
          shadows
          camera={{ position: [5, -6, 4], up: [0, 0, 1], fov: 36 }}
          dpr={[1, 1.5]}
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#c3cdd0"]} />
          <fog attach="fog" args={["#c3cdd0", 25, 70]} />
          <Lighting position={simulation.body.position} />
          <mesh receiveShadow position={[0, 0, -0.008]}>
            <planeGeometry args={[200, 200]} />
            <meshStandardMaterial color="#c3cdd0" roughness={0.95} />
          </mesh>
          <Grid
            args={[200, 200]}
            rotation={[Math.PI / 2, 0, 0]}
            position={[0, 0, -0.004]}
            cellSize={0.5}
            sectionSize={2.5}
            cellColor="#a5b6be"
            sectionColor="#8c9fa9"
            cellThickness={0.4}
            sectionThickness={0.7}
            fadeDistance={35}
            infiniteGrid
          />
          <Suspense fallback={null}>
            {simulation.body.bodies.map((pose) => {
              const leg = LEGS.indexOf(pose.name.slice(0, 2).toUpperCase());
              return (
                <Part
                  key={pose.name}
                  pose={pose}
                  activity={
                    leg < 0
                      ? 0
                      : Math.max(
                          ...simulation.body.activation.slice(
                            leg * 2,
                            leg * 2 + 2,
                          ),
                        )
                  }
                  overlay={muscles}
                  selected={selected === pose.name}
                  onSelect={() => setSelected(pose.name)}
                />
              );
            })}
          </Suspense>
          {skeleton && <Skeleton bodies={simulation.body.bodies} />}
          {simulation.body.feet.map((foot, i) => (
            <mesh key={i} position={[foot[0], foot[1], 0.007]}>
              <ringGeometry args={[0.07, 0.105, 24]} />
              <meshBasicMaterial
                color={simulation.body.foot_contacts[i] ? "#6c9c38" : "#8da2ad"}
                transparent
                opacity={0.8}
                side={THREE.DoubleSide}
              />
            </mesh>
          ))}
          <group position={simulation.environment?.source ?? [12, 3, 0.01]}>
            <mesh>
              <ringGeometry args={[0.45, 0.5, 48]} />
              <meshBasicMaterial
                color={simulation.odor === "B" ? "#9473bd" : "#829e42"}
                side={THREE.DoubleSide}
              />
            </mesh>
            <mesh position={[0, 0, 0.09]}>
              <sphereGeometry args={[0.13, 20, 12]} />
              <meshStandardMaterial color="#99b360" />
            </mesh>
          </group>
          <CameraRig
            view={view}
            follow={follow}
            position={simulation.body.position}
            heading={simulation.body.heading}
            resetKey={resetKey}
          />
        </Canvas>
        <div className="view-switch" aria-label="Camera view">
          {["fly", "top", "side"].map((v) => (
            <button
              key={v}
              className={view === v ? "active" : ""}
              onClick={() => setView(v)}
            >
              {v === "fly" ? "Perspective" : v === "top" ? "Top" : "Side"}
            </button>
          ))}
        </div>
        <div className="scene-tools">
          <button
            aria-label="Toggle muscle overlay"
            title="Effective muscle activation"
            className={muscles ? "active" : ""}
            onClick={() => setMuscles(!muscles)}
          >
            <Layers3 size={18} />
          </button>
          <button
            aria-label="Toggle skeleton and joints"
            title="Connected skeleton and joint origins"
            className={skeleton ? "active" : ""}
            onClick={() => setSkeleton(!skeleton)}
          >
            <GitBranch size={18} />
          </button>
          <button
            aria-label="Follow fly"
            title="Follow fly"
            className={follow ? "active" : ""}
            onClick={() => setFollow(!follow)}
          >
            <Focus size={18} />
          </button>
          <button
            aria-label="Reset camera"
            title="Reset camera"
            onClick={() => {
              setView("fly");
              setResetKey((n) => n + 1);
            }}
          >
            <RotateCcw size={17} />
          </button>
        </div>
        <div className="anatomy-inspector">
          <strong>{partLabel(selected)}</strong>
          <span>
            {selectedPart?.parent
              ? `Attached to ${partLabel(selectedPart.parent)}`
              : "Free body · all segments move with this root"}
          </span>
        </div>
        <div className="arena-caption">
          <Move size={13} /> Drag to orbit · select a body segment
        </div>
        <div className="arena-scale">
          <span>Grid spacing · 0.5 mm</span>
        </div>
        <div className="arena-note">
          {simulation.environment?.mode === "spatial"
            ? `${simulation.environment.distance.toFixed(1)} mm to odor · ${(100 * simulation.environment.concentration).toFixed(0)}% intensity`
            : `${anatomy.mass_mg.toFixed(2)} mg · ${anatomy.muscles} effective muscles`}
        </div>
      </div>
    </section>
  );
}
