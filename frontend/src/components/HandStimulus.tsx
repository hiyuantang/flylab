import { useEffect, useState } from "react";
import * as THREE from "three";
import type { Simulation } from "../lib/types";

export function HandStimulus({
  stimulus,
}: {
  stimulus: NonNullable<Simulation["gesture_stimulus"]>;
}) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [error, setError] = useState<Error | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let created: THREE.BufferGeometry | null = null;
    setGeometry(null);
    setError(null);
    fetch(stimulus.mesh_url, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Hand mesh could not be loaded");
        return response.json() as Promise<{
          vertices: number[];
          indices: number[];
        }>;
      })
      .then((data) => {
        if (controller.signal.aborted) return;
        created = new THREE.BufferGeometry();
        created.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(data.vertices, 3),
        );
        created.setIndex(data.indices);
        created.computeVertexNormals();
        setGeometry(created);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(error);
      });
    return () => {
      controller.abort();
      created?.dispose();
    };
  }, [stimulus.mesh_url]);
  if (error) throw error;
  if (!geometry) return null;
  return (
    <mesh
      geometry={geometry}
      position={stimulus.position}
      quaternion={
        stimulus.quaternion
          ? [
              stimulus.quaternion[1],
              stimulus.quaternion[2],
              stimulus.quaternion[3],
              stimulus.quaternion[0],
            ]
          : [0, 0, 0, 1]
      }
    >
      <meshStandardMaterial
        color={stimulus.color}
        roughness={0.65}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}
