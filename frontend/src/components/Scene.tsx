import {
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ComponentRef, ReactNode, RefObject } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Html } from "@react-three/drei/web/Html";
import { World, useWorld } from "./World";
import { HandStimulus } from "./HandStimulus";
import { Grid } from "@react-three/drei/core/Grid";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import {
  Focus,
  Layers3,
  Move,
  RotateCcw,
  GitBranch,
  ArrowLeftRight,
  Target,
} from "lucide-react";
import {
  useViewState,
  oneOf,
  readView,
  writeView,
  clearView,
} from "../lib/viewState";
import { request } from "../lib/api";
import Select from "./Select";
import { GESTURES, type GestureId } from "../lib/gestures";
import { GestureIcon } from "./GestureControls";
import { PosePlayback } from "../lib/motion";
import type {
  Simulation,
  BodyPose,
  SceneSummary,
  WorldObject,
} from "../lib/types";

const LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"];
const WARM = new THREE.Color("#c4924d");
const HOT = new THREE.Color("#ff7736");
function PlaybackClock({
  motion,
  surface,
}: {
  motion: PosePlayback;
  surface: RefObject<HTMLDivElement | null>;
}) {
  useFrame((_, delta) => {
    motion.advance(delta);
    if (surface.current) {
      surface.current.dataset.displayTime = motion.time.toFixed(6);
      surface.current.dataset.computedTime = motion.latestTime.toFixed(6);
    }
  }, -10);
  return null;
}

function BodyReady({ surface }: { surface: RefObject<HTMLDivElement | null> }) {
  useEffect(() => {
    if (surface.current) surface.current.dataset.bodyReady = "true";
  }, [surface]);
  return null;
}

function PoseTransform({
  motion,
  name,
  children,
  rotate = true,
}: {
  motion: PosePlayback;
  name: string;
  children: ReactNode;
  rotate?: boolean;
}) {
  const group = useRef<THREE.Group>(null);
  useFrame(() => {
    const pose = motion.get(name);
    if (group.current && pose) {
      group.current.position.copy(pose.position);
      if (rotate) group.current.quaternion.copy(pose.quaternion);
    }
  });
  return <group ref={group}>{children}</group>;
}
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
  motion,
}: {
  pose: NonNullable<Simulation["body"]["pretarsi"]>[number];
  overlay: boolean;
  motion: PosePlayback;
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
  return (
    <PoseTransform motion={motion} name={pose.name}>
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
    </PoseTransform>
  );
}

function Part({
  pose,
  activity,
  overlay,
  selected,
  onSelect,
  motion,
}: {
  pose: BodyPose;
  activity: number;
  overlay: boolean;
  selected: boolean;
  onSelect: () => void;
  motion: PosePlayback;
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
    <PoseTransform motion={motion} name={pose.name}>
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
    </PoseTransform>
  );
}
function Skeleton({
  bodies,
  motion,
}: {
  bodies: BodyPose[];
  motion: PosePlayback;
}) {
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
  }, [bodies.map((b) => `${b.name}:${b.parent}`).join(",")]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useFrame(() => {
    const positions = geometry.getAttribute("position");
    let i = 0;
    for (const body of bodies) {
      const parent = body.parent ? motion.get(body.parent) : undefined;
      const child = motion.get(body.name);
      if (parent && child) {
        positions.setXYZ(i++, ...parent.position.toArray());
        positions.setXYZ(i++, ...child.position.toArray());
      }
    }
    positions.needsUpdate = true;
    geometry.computeBoundingSphere();
  });
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
        <PoseTransform key={body.name} motion={motion} name={body.name}>
          <mesh renderOrder={4}>
            <sphereGeometry args={[0.025, 8, 6]} />
            <meshBasicMaterial color="#bcffff" depthTest={false} />
          </mesh>
        </PoseTransform>
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
  motion,
  stimulus,
  persistenceKey,
}: {
  view: string;
  follow: boolean;
  position: number[];
  heading: number;
  resetKey: number;
  scene: SceneSummary;
  motion: PosePlayback;
  stimulus?: Simulation["gesture_stimulus"];
  persistenceKey: string;
}) {
  const { camera, size } = useThree();
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const previous = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  const latest = useRef(position);
  const anchor = useRef(new THREE.Vector3());
  const lastReset = useRef(resetKey);
  const orbitEdited = useRef(false);
  const lastSavedOrbit = useRef("");
  const cameraKey = `${persistenceKey}.camera.${scene.id}.${view}`;
  // Keep the body and the opened wings centered as the fly turns.
  latest.current = [
    position[0] - 0.8 * Math.cos(heading),
    position[1] - 0.8 * Math.sin(heading),
    position[2],
  ];
  useEffect(() => {
    orbitEdited.current = false;
    const p = latest.current;
    const handRotation = stimulus?.quaternion
      ? new THREE.Quaternion(
          stimulus.quaternion[1],
          stimulus.quaternion[2],
          stimulus.quaternion[3],
          stimulus.quaternion[0],
        )
      : new THREE.Quaternion();
    const handCenter = new THREE.Vector3(
      ...(stimulus?.focus_offset ?? [-20, 0, 85]),
    )
      .applyQuaternion(handRotation)
      .add(new THREE.Vector3(...(stimulus?.position ?? [220, 0, 40])));
    const target =
      view === "hand"
        ? handCenter
        : view === "room"
          ? new THREE.Vector3(...scene.camera.target)
          : new THREE.Vector3(p[0], p[1], p[2] - 0.25);
    const offsets: Record<string, [number, number, number]> = {
      fly: [5.8, -7.2, 4.5],
      surface: [300, -400, 300],
      top: [0, -0.001, 10.5],
      side: [0, -9, 0.6],
      hand: [-300, -400, 170],
    };
    if (view === "hand") {
      const aspect = size.width / size.height;
      const halfAngle = Math.min(
        Math.PI / 10,
        Math.atan(Math.tan(Math.PI / 10) * aspect),
      );
      const direction = new THREE.Vector3(-1, -1.3, 0.55)
        .applyQuaternion(handRotation)
        .normalize();
      camera.position
        .copy(target)
        .addScaledVector(
          direction,
          ((stimulus?.framing_radius ?? 200) / Math.sin(halfAngle)) * 1.08,
        );
    } else if (view === "room") {
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
    anchor.current.copy(target);
    if (lastReset.current !== resetKey) clearView(cameraKey);
    lastReset.current = resetKey;
    const saved = readView(
      cameraKey,
      { offset: [] as number[], targetOffset: [] as number[] },
      (value): value is { offset: number[]; targetOffset: number[] } => {
        if (!value || typeof value !== "object") return false;
        const v = value as { offset?: unknown; targetOffset?: unknown };
        return [v.offset, v.targetOffset].every(
          (x) => Array.isArray(x) && x.length === 3 && x.every(Number.isFinite),
        );
      },
    );
    if (saved.offset.length === 3) {
      target.add(new THREE.Vector3(...saved.targetOffset));
      camera.position.copy(target).add(new THREE.Vector3(...saved.offset));
    }
    camera.up.set(0, 0, 1);
    controls.current?.target.copy(target);
    camera.lookAt(target);
    previous.current.set(...(p as [number, number, number]));
    controls.current?.update();
  }, [
    view,
    cameraKey,
    camera,
    resetKey,
    scene.id,
    size.width,
    size.height,
    stimulus?.gesture,
    stimulus?.position.join(","),
    stimulus?.quaternion?.join(","),
  ]);
  useFrame(() => {
    const root = motion.get("c_thorax");
    if (root) {
      direction.current.set(1, 0, 0).applyQuaternion(root.quaternion);
      const displayedHeading = Math.atan2(
        direction.current.y,
        direction.current.x,
      );
      latest.current[0] = root.position.x - 0.8 * Math.cos(displayedHeading);
      latest.current[1] = root.position.y - 0.8 * Math.sin(displayedHeading);
      latest.current[2] = root.position.z;
    }
    const p = latest.current;
    if (follow && view !== "room" && view !== "hand") {
      const dx = p[0] - previous.current.x,
        dy = p[1] - previous.current.y,
        dz = p[2] - previous.current.z;
      anchor.current.add(new THREE.Vector3(dx, dy, dz));
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
      onStart={() => {
        orbitEdited.current = true;
      }}
      onChange={() => {
        if (!orbitEdited.current || !controls.current) return;
        // Orbit damping continues after pointer-up. Save its final position too.
        const rounded = (v: THREE.Vector3) =>
          v.toArray().map((x) => Math.round(x * 1e6) / 1e6);
        const state = {
          offset: rounded(camera.position.clone().sub(controls.current.target)),
          targetOffset: rounded(
            controls.current.target.clone().sub(anchor.current),
          ),
        };
        const encoded = JSON.stringify(state);
        if (encoded !== lastSavedOrbit.current) {
          lastSavedOrbit.current = encoded;
          writeView(cameraKey, state);
        }
      }}
      makeDefault
      maxPolarAngle={Math.PI * 0.49}
      minDistance={2.4}
      maxDistance={
        view === "room"
          ? 22000
          : view === "surface" || view === "hand"
            ? 1500
            : 70
      }
    />
  );
}
function Lighting({
  position,
  illumination,
  view,
  scene,
  motion,
}: {
  position: number[];
  illumination: number;
  view: string;
  scene: SceneSummary;
  motion: PosePlayback;
}) {
  const room = view === "room";
  const radius = room
    ? Math.max(...scene.extent)
    : view === "surface"
      ? 350
      : 5;
  const target = useMemo(() => new THREE.Object3D(), []);
  const light = useRef<THREE.DirectionalLight>(null);
  target.position.set(
    ...((room ? scene.camera.target : position) as [number, number, number]),
  );
  useFrame(() => {
    const root = motion.get("c_thorax");
    if (!room && root && light.current) {
      target.position.copy(root.position);
      light.current.position.set(
        root.position.x + radius * 0.8,
        root.position.y - radius,
        root.position.z + radius * 1.8,
      );
    }
  });
  return (
    <>
      <ambientLight intensity={0.7 * illumination} />
      <hemisphereLight args={["#fff8e9", "#767668", 1.6 * illumination]} />
      <primitive object={target} />
      <directionalLight
        ref={light}
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
function SceneView({
  simulation,
  gestureControls,
  inset = false,
  persistenceKey = "experiment",
  highlightMuscles = false,
  heading,
  sceneOverlay,
  sceneFooter,
  targetPreview = false,
}: {
  simulation: Simulation;
  gestureControls?: ReactNode;
  inset?: boolean;
  persistenceKey?: string;
  highlightMuscles?: boolean;
  heading?: { title: string; subtitle: string; control?: ReactNode };
  sceneOverlay?: ReactNode;
  sceneFooter?: ReactNode;
  targetPreview?: boolean;
}) {
  const motion = useMemo(() => {
    const playback = new PosePlayback();
    playback.push(simulation, performance.now() / 1000);
    return playback;
  }, []);
  const surface = useRef<HTMLDivElement>(null);
  useLayoutEffect(
    () => motion.push(simulation, performance.now() / 1000),
    [motion, simulation],
  );
  const scene = simulation.scene;
  const room = scene.id !== "lab";
  const { definition, error, retry } = useWorld(scene);
  const [storedView, setView] = useViewState(
    `${persistenceKey}.view`,
    simulation.gesture_stimulus && !inset ? "hand" : room ? "room" : "fly",
    oneOf(["hand", "room", "surface", "fly", "top", "side"]),
  );
  const view =
    (inset || !simulation.gesture_stimulus) && storedView === "hand"
      ? "fly"
      : !room && ["room", "surface"].includes(storedView)
        ? "fly"
        : storedView;
  const cueKey = `${simulation.gesture_stimulus?.gesture}:${simulation.gesture_stimulus?.position.join(",")}`;
  const previousCue = useRef(cueKey);
  useEffect(() => {
    if (inset || previousCue.current === cueKey) return;
    previousCue.current = cueKey;
    if (simulation.gesture_stimulus) setView("hand");
    else setView((current) => (current === "hand" ? "fly" : current));
  }, [cueKey, inset]);
  const [object, setObject] = useState<WorldObject | null>(null);
  const selectObject = useCallback(
    (value: WorldObject) => setObject(value),
    [],
  );
  const [muscles, setMuscles] = useViewState(
    `${persistenceKey}.muscles`,
    false,
  );
  const [skeleton, setSkeleton] = useViewState(
    `${persistenceKey}.skeleton`,
    false,
  );
  const [follow, setFollow] = useViewState(`${persistenceKey}.follow`, true);
  const [selected, setSelected] = useViewState(
    `${persistenceKey}.selected`,
    "c_thorax",
  );
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
    <section
      className={`panel arena-panel ${targetPreview ? "target-scene" : ""} ${inset ? "scene-inset" : ""} ${simulation.gesture_stimulus && view === "hand" ? "hand-overview" : ""}`}
    >
      <div className="panel-heading arena-heading">
        <div>
          <h2>{heading?.title ?? scene.title}</h2>
          <p>
            {heading?.subtitle ?? (
              <>
                {scene.spawn_label} ·{" "}
                {room
                  ? `${scene.extent[0] / 1000} × ${scene.extent[1] / 1000} m`
                  : "Controlled arena"}
              </>
            )}
          </p>
        </div>
        {heading?.control ?? (
          <span
            className="model-tag"
            title="NeuroMechFly research anatomy, derived from a female specimen"
          >
            {room ? "True scale" : "Research anatomy"}
          </span>
        )}
      </div>
      <div className="scene-wrap" ref={surface}>
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
          onPointerMissed={(event) => {
            if (event.type === "click") {
              setSelected("");
              setObject(null);
            }
          }}
        >
          <PlaybackClock motion={motion} surface={surface} />
          <color attach="background" args={[background]} />
          {!room && !simulation.gesture_stimulus && (
            <fog attach="fog" args={[background, 25, 70]} />
          )}
          {simulation.gesture_stimulus && !inset && (
            <HandStimulus stimulus={simulation.gesture_stimulus} />
          )}
          <Lighting
            position={simulation.body.position}
            illumination={illumination}
            view={view}
            scene={scene}
            motion={motion}
          />
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
            <PoseTransform motion={motion} name="c_thorax" rotate={false}>
              <group position={[0, 0, 160]}>
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
            </PoseTransform>
          )}
          <Suspense
            fallback={
              inset || view === "fly" ? (
                <Html center>
                  <div className="body-loading">Loading fly anatomy…</div>
                </Html>
              ) : null
            }
          >
            {simulation.body.bodies.map((pose) => {
              const leg = LEGS.indexOf(pose.name.slice(0, 2).toUpperCase());
              return (
                <Part
                  key={pose.name}
                  pose={pose}
                  motion={motion}
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
                  overlay={muscles || highlightMuscles}
                  selected={selected === pose.name}
                  onSelect={() => {
                    setSelected(pose.name);
                    setObject(null);
                  }}
                />
              );
            })}
            <BodyReady surface={surface} />
          </Suspense>
          {simulation.body.pretarsi?.map((pose) => (
            <Pretarsus
              key={pose.name}
              pose={pose}
              overlay={muscles || highlightMuscles}
              motion={motion}
            />
          ))}
          {simulation.body.additional_geometry?.map((pose) => (
            <Pretarsus
              key={pose.name}
              pose={pose}
              overlay={muscles || highlightMuscles}
              motion={motion}
            />
          ))}
          {skeleton && (
            <Skeleton bodies={simulation.body.bodies} motion={motion} />
          )}
          {simulation.body.feet.map((_, i) => (
            <PoseTransform
              key={i}
              motion={motion}
              name={`foot:${i}`}
              rotate={false}
            >
              <mesh position={[0, 0, 0.007]}>
                <ringGeometry args={[0.07, 0.105, 24]} />
                <meshBasicMaterial
                  color={
                    simulation.body.foot_contacts[i] ? "#6c9c38" : "#8da2ad"
                  }
                  transparent
                  opacity={0.8}
                  side={THREE.DoubleSide}
                />
              </mesh>
            </PoseTransform>
          ))}
          {room && (
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
          )}
          <CameraRig
            persistenceKey={persistenceKey}
            stimulus={simulation.gesture_stimulus}
            view={view}
            follow={follow}
            position={simulation.body.position}
            heading={simulation.body.heading}
            resetKey={resetKey}
            scene={scene}
            motion={motion}
          />
        </Canvas>
        <div className="view-switch" aria-label="Camera view">
          {(room
            ? ["room", "surface", "fly", "top"]
            : simulation.gesture_stimulus
              ? ["fly", "hand", "top", "side"]
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
                  hand: "Hand & lab",
                }[v]
              }
            </button>
          ))}
        </div>
        {!inset && simulation.gesture_stimulus && view === "hand" && (
          <div className="fly-response-inset" data-response-mode="current">
            <div className="fly-response-heading">
              <span>Fly response</span>
            </div>
            <SceneView
              simulation={simulation}
              inset
              persistenceKey={`${persistenceKey}.response-camera`}
            />
          </div>
        )}
        {sceneOverlay}
        <div className="scene-tools">
          {gestureControls}
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
              clearView(`${persistenceKey}.camera.${scene.id}.${view}`);
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
        {(object || selected) && (
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
        )}
        <div className="arena-caption">
          <Move size={13} /> Drag to orbit · scroll to zoom · smooth slow-motion
          playback
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
      {sceneFooter}
    </section>
  );
}

type TargetResponse = {
  body: Simulation["body"];
  steps: number;
  time: number;
  selected_muscles: number[];
  validation?: {
    pose_reached: boolean;
    checked_seconds: number;
    foot_heights_mm: number[];
    body_height_mm: number;
  };
};
export function FlyScene({
  simulation,
  gestureControls,
  persistenceKey = "experiment",
  targetSteps = 10,
  allowTargets = false,
}: {
  simulation: Simulation;
  gestureControls?: ReactNode;
  persistenceKey?: string;
  targetSteps?: number;
  allowTargets?: boolean;
}) {
  const [storedMode, setMode] = useViewState(
    `${persistenceKey}.scene-mode`,
    "result",
    oneOf(["result", "target"]),
  );
  const mode =
    allowTargets && simulation.scene.id === "lab" ? storedMode : "result";
  const [gesture, setGesture] = useViewState<GestureId>(
    `${persistenceKey}.target-gesture`,
    simulation.gesture_stimulus?.gesture ?? "palm",
    oneOf(GESTURES.map((g) => g.id)),
  );
  const [result, setResult] = useState<{
    key: string;
    value?: TargetResponse;
    error?: string;
  } | null>(null);
  const parameters = simulation.body_parameters;
  const validSteps =
    Number.isInteger(targetSteps) && targetSteps >= 1 && targetSteps <= 250;
  const requestKey = JSON.stringify({
    gesture,
    steps: targetSteps,
    parameters,
  });
  const cacheKey = `routed-muscle-reference-v3:${requestKey}`;
  useEffect(() => {
    if (mode !== "target" || !parameters || !validSteps) return;
    let disposed = false;
    request<TargetResponse>("/gestures/target", JSON.parse(requestKey))
      .then((value) => {
        if (!disposed) setResult({ key: cacheKey, value });
      })
      .catch((error) => {
        if (!disposed) setResult({ key: cacheKey, error: error.message });
      });
    return () => {
      disposed = true;
    };
  }, [mode, cacheKey]);
  const target = result?.key === cacheKey ? result.value : undefined;
  const error = result?.key === cacheKey ? result.error : undefined;
  const cue = GESTURES.find((g) => g.id === gesture)!;
  const reference: Simulation = {
    ...simulation,
    body: target?.body ?? simulation.body,
    time: target?.time ?? 0,
    steps: target?.steps ?? 0,
    running: false,
    gesture_stimulus: null,
  };
  const control = (
    <button
      className="scene-target-toggle"
      disabled={simulation.scene.id !== "lab"}
      title={
        simulation.scene.id !== "lab"
          ? "Target previews use the laboratory"
          : undefined
      }
      aria-pressed={mode === "target"}
      onClick={() => setMode(mode === "target" ? "result" : "target")}
    >
      {mode === "target" ? <ArrowLeftRight size={15} /> : <Target size={15} />}
      {mode === "target" ? "Back to result" : "View targets"}
    </button>
  );
  return (
    <SceneView
      key={mode === "target" ? `target:${requestKey}:${!!target}` : "result"}
      simulation={mode === "target" ? reference : simulation}
      persistenceKey={
        mode === "target" ? `${persistenceKey}.target` : persistenceKey
      }
      gestureControls={mode === "target" ? undefined : gestureControls}
      highlightMuscles={mode === "target"}
      targetPreview={mode === "target"}
      heading={{
        title:
          mode === "target" ? "Target muscle preview" : simulation.scene.title,
        subtitle:
          mode === "target"
            ? `Reference physics · ${(targetSteps * 0.02).toFixed(2)} s`
            : `${simulation.scene.spawn_label} · ${simulation.scene.id === "lab" ? "Controlled arena" : "True scale"}`,
        control: allowTargets ? control : undefined,
      }}
      sceneOverlay={
        mode === "target" && !target ? (
          <div className="target-scene-pending" role="status">
            {error ??
              (!parameters
                ? "Target mechanics unavailable for this preview"
                : !validSteps
                  ? "Enter 1–250 steps to preview the target"
                  : "Building target response…")}
          </div>
        ) : undefined
      }
      sceneFooter={
        mode === "target" ? (
          <div className="target-gesture-picker">
            <label>
              <GestureIcon gesture={gesture} />
              <span>Gesture input</span>
              <Select
                aria-label="Target gesture"
                value={gesture}
                onChange={(value) => setGesture(value as GestureId)}
                options={GESTURES.map((g) => ({ value: g.id, label: g.name }))}
              />
            </label>
            <p>Intended: {cue.target.toLowerCase()}</p>
            <small>
              {target?.validation?.pose_reached
                ? "Pose verified in physics · includes stance muscles"
                : "Reference pose not verified at this duration"}
            </small>
          </div>
        ) : undefined
      }
    />
  );
}
