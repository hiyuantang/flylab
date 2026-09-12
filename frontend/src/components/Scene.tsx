import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ComponentRef } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Html } from "@react-three/drei/web/Html";
import { World, useWorld } from "./World";
import { Grid } from "@react-three/drei/core/Grid";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { Focus, Layers3, Move, RotateCcw, GitBranch } from "lucide-react";
import type {
  Simulation,
  BodyPose,
  SceneSummary,
  WorldObject,
} from "../lib/types";

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
function Pretarsus({
  pose,
  overlay,
}: {
  pose: NonNullable<Simulation["body"]["pretarsi"]>[number];
  overlay: boolean;
}) {
  const shapes = useMemo(
    () =>
      pose.capsules.map((c) => {
        const a = new THREE.Vector3(
          ...(c.slice(0, 3) as [number, number, number]),
        );
        const b = new THREE.Vector3(
          ...(c.slice(3, 6) as [number, number, number]),
        );
        const direction = b.clone().sub(a);
        return {
          position: a.add(b).multiplyScalar(0.5),
          length: direction.length(),
          quaternion: new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize(),
          ),
          radius: c[6],
        };
      }),
    [JSON.stringify(pose.capsules)],
  );
  const [w, x, y, z] = pose.quaternion;
  return (
    <group position={pose.position} quaternion={[x, y, z, w]}>
      {shapes.map((shape, index) => (
        <mesh
          key={index}
          position={shape.position}
          quaternion={shape.quaternion}
        >
          <capsuleGeometry args={[shape.radius, shape.length, 4, 6]} />
          <meshStandardMaterial
            color={
              overlay ? WARM.clone().lerp(HOT, pose.activation) : "#553a22"
            }
            roughness={0.7}
          />
        </mesh>
      ))}
    </group>
  );
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
  scene,
}: {
  view: string;
  follow: boolean;
  position: number[];
  heading: number;
  resetKey: number;
  scene: SceneSummary;
}) {
  const { camera, size } = useThree();
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
    const target =
      view === "room"
        ? new THREE.Vector3(...scene.camera.target)
        : new THREE.Vector3(p[0], p[1], p[2] - 0.25);
    const offsets: Record<string, [number, number, number]> = {
      fly: [5.8, -7.2, 4.5],
      surface: [300, -400, 300],
      top: [0, -0.001, 10.5],
      side: [0, -9, 0.6],
    };
    if (view === "room") {
      const direction = new THREE.Vector3(...scene.camera.position)
        .sub(target)
        .normalize();
      const radius = Math.hypot(...scene.extent) / 2;
      const aspect = size.width / size.height;
      const angle = Math.min(
        (36 * Math.PI) / 360,
        Math.atan(Math.tan((36 * Math.PI) / 360) * aspect),
      );
      camera.position
        .copy(target)
        .addScaledVector(direction, (radius / Math.sin(angle)) * 1.12);
    } else
      camera.position.copy(target).add(new THREE.Vector3(...offsets[view]));
    camera.up.set(0, 0, 1);
    controls.current?.target.copy(target);
    camera.lookAt(target);
    previous.current.set(...(p as [number, number, number]));
    controls.current?.update();
  }, [view, camera, resetKey, scene.id, size.width, size.height]);
  useFrame(() => {
    const p = latest.current;
    if (follow && view !== "room") {
      const dx = p[0] - previous.current.x,
        dy = p[1] - previous.current.y,
        dz = p[2] - previous.current.z;
      camera.position.x += dx;
      camera.position.y += dy;
      camera.position.z += dz;
      if (controls.current) {
        controls.current.target.x += dx;
        controls.current.target.y += dy;
        controls.current.target.z += dz;
      }
    }
    previous.current.set(...(p as [number, number, number]));
  });
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      maxPolarAngle={Math.PI * 0.49}
      minDistance={2.4}
      maxDistance={view === "room" ? 22000 : view === "surface" ? 1500 : 70}
    />
  );
}
function Lighting({
  position,
  illumination,
  view,
  scene,
}: {
  position: number[];
  illumination: number;
  view: string;
  scene: SceneSummary;
}) {
  const room = view === "room";
  const radius = room
    ? Math.max(...scene.extent)
    : view === "surface"
      ? 350
      : 5;
  const target = useMemo(() => new THREE.Object3D(), []);
  target.position.set(
    ...((room ? scene.camera.target : position) as [number, number, number]),
  );
  return (
    <>
      <ambientLight intensity={0.7 * illumination} />
      <hemisphereLight args={["#fff8e9", "#767668", 1.6 * illumination]} />
      <primitive object={target} />
      <directionalLight
        target={target}
        position={[
          target.position.x + radius * 0.8,
          target.position.y - radius,
          target.position.z + radius * 1.8,
        ]}
        intensity={2.4 * illumination}
        castShadow
        shadow-mapSize={room ? [4096, 4096] : [2048, 2048]}
        shadow-camera-left={-radius}
        shadow-camera-right={radius}
        shadow-camera-top={radius}
        shadow-camera-bottom={-radius}
        shadow-camera-near={0.01}
        shadow-camera-far={radius * 5}
        shadow-normalBias={radius * 0.0006}
        shadow-bias={-0.00002}
      />
    </>
  );
}
export function FlyScene({ simulation }: { simulation: Simulation }) {
  const scene = simulation.scene;
  const room = scene.id !== "lab";
  const { definition, error, retry } = useWorld(scene);
  const [view, setView] = useState(room ? "room" : "fly");
  const [object, setObject] = useState<WorldObject | null>(null);
  const selectObject = useCallback(
    (value: WorldObject) => setObject(value),
    [],
  );
  const [muscles, setMuscles] = useState(false);
  const [skeleton, setSkeleton] = useState(false);
  const [follow, setFollow] = useState(true);
  const [selected, setSelected] = useState("c_thorax");
  const [resetKey, setResetKey] = useState(0);
  const selectedPart = simulation.body.bodies.find(
    (body) => body.name === selected,
  );
  const anatomy = simulation.body.anatomy;
  const illumination = simulation.senses?.settings.illumination ?? 1;
  const background = useMemo(
    () => new THREE.Color(scene.background).multiplyScalar(illumination),
    [illumination, scene.background],
  );
  return (
    <section className="panel arena-panel">
      <div className="panel-heading arena-heading">
        <div>
          <h2>{scene.title}</h2>
          <p>
            {scene.spawn_label} ·{" "}
            {room
              ? `${scene.extent[0] / 1000} × ${scene.extent[1] / 1000} m`
              : "Controlled arena"}
          </p>
        </div>
        <span
          className="model-tag"
          title="NeuroMechFly research anatomy, derived from a female specimen"
        >
          {room ? "True scale" : "Research anatomy"}
        </span>
      </div>
      <div className="scene-wrap">
        <Canvas
          shadows
          camera={{
            position: [5, -6, 4],
            up: [0, 0, 1],
            fov: 36,
            near: 0.02,
            far: 30000,
          }}
          dpr={[1, 1.5]}
          gl={{ antialias: true, logarithmicDepthBuffer: true }}
        >
          <color attach="background" args={[background]} />
          {!room && <fog attach="fog" args={[background, 25, 70]} />}
          <Lighting
            position={simulation.body.position}
            illumination={illumination}
            view={view}
            scene={scene}
          />
          {!room && simulation.senses && (
            <mesh position={simulation.senses.settings.stimulus_position}>
              <sphereGeometry args={[0.7, 32, 20]} />
              <meshBasicMaterial color="#101010" />
            </mesh>
          )}
          {!room && (
            <>
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
            </>
          )}
          {room && definition && (
            <World
              definition={definition}
              selected={object?.id ?? null}
              onSelect={selectObject}
            />
          )}
          {room && view === "room" && (
            <group
              position={[
                simulation.body.position[0],
                simulation.body.position[1],
                simulation.body.position[2] + 160,
              ]}
            >
              <mesh renderOrder={10} rotation={[-Math.PI / 2, 0, 0]}>
                <coneGeometry args={[35, 100, 12]} />
                <meshBasicMaterial color="#d9f96e" depthTest={false} />
              </mesh>
              <Html center position={[0, 0, 130]} zIndexRange={[10, 0]}>
                <button
                  className="fly-location"
                  onClick={() => {
                    setView("fly");
                    setObject(null);
                  }}
                >
                  Fly here <Focus size={12} />
                </button>
              </Html>
            </group>
          )}
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
                  onSelect={() => {
                    setSelected(pose.name);
                    setObject(null);
                  }}
                />
              );
            })}
          </Suspense>
          {simulation.body.pretarsi?.map((pose) => (
            <Pretarsus key={pose.name} pose={pose} overlay={muscles} />
          ))}
          {simulation.body.additional_geometry?.map((pose) => (
            <Pretarsus key={pose.name} pose={pose} overlay={muscles} />
          ))}
          {skeleton && <Skeleton bodies={simulation.body.bodies} />}
          {simulation.body.feet.map((foot, i) => (
            <mesh key={i} position={[foot[0], foot[1], foot[2] + 0.007]}>
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
            scene={scene}
          />
        </Canvas>
        <div className="view-switch" aria-label="Camera view">
          {(room
            ? ["room", "surface", "fly", "top"]
            : ["fly", "top", "side"]
          ).map((v) => (
            <button
              key={v}
              className={view === v ? "active" : ""}
              onClick={() => {
                setView(v);
                setObject(null);
              }}
            >
              {
                {
                  room: "Overview",
                  surface: "Surface",
                  fly: "Fly view",
                  top: "Top",
                  side: "Side",
                }[v]
              }
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
        {(error || (room && !definition)) && (
          <div className="world-loading" role="status">
            {error ?? "Loading scene geometry…"}
            {error && <button onClick={retry}>Retry</button>}
          </div>
        )}
        <div className="anatomy-inspector">
          <strong>
            {object
              ? object.label
              : view === "room" || view === "surface"
                ? scene.spawn_label
                : partLabel(selected)}
          </strong>
          <span>
            {object
              ? `${object.category} · ${object.shape} · ${object.size.map((n) => (n * 2).toLocaleString(undefined, { maximumFractionDigits: 1 })).join(" × ")} mm`
              : view === "room"
                ? `${scene.solid_count} solids · select an object to inspect`
                : view === "surface"
                  ? "Nearby objects at the fly’s starting surface"
                  : selectedPart?.parent
                    ? `Attached to ${partLabel(selectedPart.parent)}`
                    : "Free body · all segments move with this root"}
          </span>
        </div>
        <div className="arena-caption">
          <Move size={13} /> Drag to orbit · scroll to zoom
        </div>
        <div className="arena-scale">
          <span>
            {room
              ? view === "room"
                ? "Life-size room · marker locates the 2–3 mm fly"
                : "Millimetre-scale fly · full-size surroundings"
              : "Grid spacing · 0.5 mm"}
          </span>
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
