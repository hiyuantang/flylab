import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei/core/OrbitControls";
import { Grid } from "@react-three/drei/core/Grid";
import { ContactShadows } from "@react-three/drei/core/ContactShadows";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Focus, Layers3, Move, RotateCcw } from "lucide-react";
import type { Simulation, BodyPose } from "../lib/types";

function ResearchMesh({
  name,
  size,
  color,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  opacity = 1,
}: {
  name: string;
  size: [number, number, number];
  color: string;
  position?: [number, number, number];
  rotation?: [number, number, number];
  opacity?: number;
}) {
  const source = useLoader(STLLoader, `/models/${name}.stl`);
  const geometry = useMemo(() => {
    const g = source.clone();
    g.computeBoundingBox();
    const b = g.boundingBox!;
    const extent = b.getSize(new THREE.Vector3());
    g.center();
    g.scale(size[0] / extent.x, size[1] / extent.y, size[2] / extent.z);
    g.computeVertexNormals();
    return g;
  }, [source, size[0], size[1], size[2]]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh
      geometry={geometry}
      position={position}
      rotation={rotation}
      castShadow
    >
      <meshStandardMaterial
        color={color}
        roughness={0.62}
        metalness={0.08}
        transparent={opacity < 1}
        opacity={opacity}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}
function Capsule({
  end,
  radius = 0.025,
  color = "#7b5830",
}: {
  end: number[];
  radius?: number;
  color?: string;
}) {
  const { mid, q, len } = useMemo(() => {
    const e = new THREE.Vector3(...(end as [number, number, number]));
    return {
      mid: e.clone().multiplyScalar(0.5),
      q: new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        e.clone().normalize(),
      ),
      len: e.length(),
    };
  }, [end[0], end[1], end[2]]);
  return (
    <mesh position={mid} quaternion={q} castShadow>
      <capsuleGeometry args={[radius, len, 5, 10]} />
      <meshStandardMaterial color={color} roughness={0.62} />
    </mesh>
  );
}
function Part({
  pose,
  activation,
  showMuscles,
}: {
  pose: BodyPose;
  activation: number[];
  showMuscles: boolean;
}) {
  const group = useRef<THREE.Group>(null);
  useFrame(() => {
    if (group.current) {
      group.current.position.set(...pose.position);
      group.current.quaternion.set(
        pose.quaternion[1],
        pose.quaternion[2],
        pose.quaternion[3],
        pose.quaternion[0],
      );
    }
  });
  const i = ["LF", "LM", "LH", "RF", "RM", "RH"].indexOf(pose.name.slice(0, 2));
  const side = i < 3 ? 1 : -1;
  const dx = [0.26, 0, -0.28][i % 3];
  const a =
    i < 0 ? 0 : Math.max(activation[i * 2] ?? 0, activation[i * 2 + 1] ?? 0);
  const muscle = new THREE.Color("#5f4830").lerp(new THREE.Color("#ff9c57"), a);
  return (
    <group ref={group}>
      {pose.name === "thorax" && (
        <>
          <ResearchMesh
            name="c_thorax"
            size={[0.76, 0.46, 0.4]}
            color="#87613c"
          />
          {[-1, 1].map((s) => (
            <group
              key={s}
              position={[-0.05, s * 0.14, 0.15]}
              rotation={[s * 0.1, 0.08, s * 0.18]}
            >
              <ResearchMesh
                name="l_wing"
                size={[0.4, 1.48, 0.035]}
                position={[-0.47, s * 0.18, 0.075]}
                rotation={[0, 0, Math.PI / 2]}
                color="#dddccb"
                opacity={0.38}
              />
              {[0.0, 0.07, -0.07].map((v, k) => (
                <mesh
                  key={k}
                  position={[-0.42, s * 0.17 + v, 0.087]}
                  scale={[0.64, 0.003, 0.003]}
                >
                  <sphereGeometry args={[1, 16, 8]} />
                  <meshStandardMaterial
                    color="#947c57"
                    transparent
                    opacity={0.5}
                  />
                </mesh>
              ))}
            </group>
          ))}
          {Array.from({ length: 15 }, (_, j) => (
            <Capsule
              key={j}
              radius={0.0025}
              color="#2f271e"
              end={[
                0.04 + Math.sin(j) * 0.045,
                Math.cos(j) * 0.23,
                0.19 + Math.sin(j * 3) * 0.07,
              ]}
            />
          ))}
        </>
      )}
      {pose.name === "abdomen" && (
        <>
          <ResearchMesh
            name="c_abdomen12"
            size={[0.43, 0.39, 0.33]}
            position={[0.14, 0, 0]}
            color="#8c7044"
          />
          <ResearchMesh
            name="c_abdomen3"
            size={[0.19, 0.4, 0.33]}
            position={[-0.04, 0, 0]}
            color="#4d3b26"
          />
          <ResearchMesh
            name="c_abdomen4"
            size={[0.18, 0.34, 0.29]}
            position={[-0.16, 0, 0]}
            color="#947847"
          />
          <ResearchMesh
            name="c_abdomen5"
            size={[0.17, 0.25, 0.23]}
            position={[-0.27, 0, 0]}
            color="#453323"
          />
          <ResearchMesh
            name="c_abdomen6"
            size={[0.15, 0.17, 0.17]}
            position={[-0.36, 0, 0]}
            color="#7d623e"
          />
        </>
      )}
      {pose.name === "head" && (
        <>
          <ResearchMesh
            name="c_head"
            size={[0.42, 0.46, 0.38]}
            color="#a78353"
          />
          {[-1, 1].map((s) => (
            <group key={s}>
              <ResearchMesh
                name="l_eye"
                size={[0.23, 0.14, 0.28]}
                position={[0.07, s * 0.19, 0.015]}
                rotation={[s === 1 ? 0 : Math.PI, 0, 0]}
                color="#8f302e"
              />
              <group position={[0.16, s * 0.07, 0.1]}>
                <Capsule
                  end={[0.13, s * 0.05, 0.07]}
                  radius={0.012}
                  color="#4a3925"
                />
                <group position={[0.13, s * 0.05, 0.07]}>
                  <Capsule end={[0.1, s * 0.04, 0.03]} radius={0.003} />
                </group>
              </group>
            </group>
          ))}
        </>
      )}
      {pose.name.endsWith("_upper") && (
        <>
          <Capsule
            end={[dx, side * 0.43, -0.24]}
            radius={0.028}
            color={showMuscles ? muscle.getStyle() : "#8b663c"}
          />
          {showMuscles && (
            <Capsule
              end={[dx * 0.9, side * 0.38, -0.2]}
              radius={0.046}
              color={muscle.getStyle()}
            />
          )}
        </>
      )}
      {pose.name.endsWith("_lower") && (
        <>
          <Capsule end={[dx * 0.5, side * 0.17, -0.43]} radius={0.017} />
          <group position={[dx * 0.5, side * 0.17, -0.43]}>
            <Capsule end={[0.12, side * 0.08, 0]} radius={0.009} />
          </group>
        </>
      )}
    </group>
  );
}
function CameraView({
  view,
  follow,
  position,
}: {
  view: string;
  follow: boolean;
  position: number[];
}) {
  const { camera } = useThree();
  const previous = useRef(new THREE.Vector3());
  useEffect(() => {
    const positions: Record<string, number[]> = {
      fly: [3.2, -4, 2.7],
      top: [0, 0, 6],
      side: [0, -5, 1.3],
    };
    camera.position.set(...(positions[view] as [number, number, number]));
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0.5);
    previous.current.set(0, 0, 0);
  }, [view, camera]);
  useFrame(() => {
    if (follow) {
      const now = new THREE.Vector3(position[0], position[1], 0);
      camera.position.add(now.clone().sub(previous.current));
      previous.current.copy(now);
    }
  });
  return null;
}
export function FlyScene({ simulation }: { simulation: Simulation }) {
  const [view, setView] = useState("fly");
  const [muscles, setMuscles] = useState(true);
  const [follow, setFollow] = useState(false);
  return (
    <section className="panel arena-panel">
      <div className="panel-heading arena-heading">
        <div>
          <h2>Embodied simulation</h2>
          <p>Six legs. One connected experiment.</p>
        </div>
        <span className="model-tag">Schematic body</span>
      </div>
      <div className="scene-wrap">
        <Canvas
          shadows
          camera={{ position: [3.2, -4, 2.7], up: [0, 0, 1], fov: 36 }}
          dpr={[1, 1.5]}
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#bfc9cd"]} />
          <fog attach="fog" args={["#bfc9cd", 12, 28]} />
          <ambientLight intensity={1.4} />
          <directionalLight
            position={[3, -4, 7]}
            intensity={3}
            castShadow
            shadow-mapSize={[2048, 2048]}
            shadow-camera-left={-5}
            shadow-camera-right={5}
            shadow-camera-top={5}
            shadow-camera-bottom={-5}
            shadow-bias={-0.0002}
          />
          <Suspense fallback={null}>
            <mesh receiveShadow position={[0, 0, -0.012]}>
              <planeGeometry args={[100, 100]} />
              <meshStandardMaterial color="#bcc7cc" roughness={0.85} />
            </mesh>
            <Grid
              args={[50, 50]}
              rotation={[Math.PI / 2, 0, 0]}
              position={[0, 0, -0.008]}
              cellSize={0.5}
              sectionSize={2.5}
              cellColor="#a3b3ba"
              sectionColor="#8fa3ac"
              cellThickness={0.4}
              sectionThickness={0.65}
              fadeDistance={20}
              infiniteGrid
            />
            {simulation.body.bodies.map((pose) => (
              <Part
                key={pose.name}
                pose={pose}
                activation={simulation.body.activation}
                showMuscles={muscles}
              />
            ))}
            <group position={[2.4, 1.2, 0.006]}>
              <mesh>
                <ringGeometry args={[0.29, 0.31, 64]} />
                <meshBasicMaterial
                  color={simulation.odor === "B" ? "#b591de" : "#d6ee9b"}
                  side={THREE.DoubleSide}
                />
              </mesh>
              <mesh>
                <ringGeometry args={[0.43, 0.44, 64]} />
                <meshBasicMaterial color="#d6ee9b" transparent opacity={0.5} />
              </mesh>
              <mesh position={[0, 0, 0.06]}>
                <sphereGeometry args={[0.09, 24, 16]} />
                <meshStandardMaterial
                  color="#d6ee9b"
                  emissive="#9cb967"
                  emissiveIntensity={0.3}
                />
              </mesh>
            </group>
            <ContactShadows
              rotation={[Math.PI / 2, 0, 0]}
              position={[0, 0, 0.002]}
              opacity={0.25}
              scale={12}
              blur={2}
              far={2}
            />
          </Suspense>
          <OrbitControls
            makeDefault
            target={[
              simulation.body.position[0] * (follow ? 1 : 0),
              simulation.body.position[1] * (follow ? 1 : 0),
              0.45,
            ]}
            maxPolarAngle={Math.PI * 0.48}
            minDistance={1.8}
            maxDistance={15}
          />
          <CameraView
            view={view}
            follow={follow}
            position={simulation.body.position}
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
            title="Muscle overlay"
            className={muscles ? "active" : ""}
            onClick={() => setMuscles(!muscles)}
          >
            <Layers3 size={18} />
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
              setFollow(false);
              setView(view === "fly" ? "top" : "fly");
            }}
          >
            <RotateCcw size={17} />
          </button>
        </div>
        <div className="arena-caption">
          <Move size={13} /> Drag to orbit · scroll to zoom
        </div>
        <div className="arena-scale">
          <span>Uncalibrated scale</span>
          <div />
        </div>
        <div className="arena-note">Muscle-driven · MuJoCo physics</div>
      </div>
    </section>
  );
}
