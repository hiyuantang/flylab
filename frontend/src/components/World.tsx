import { memo, useEffect, useState } from "react";
import {
  FlaskConical,
  CookingPot,
  Sofa,
  BedDouble,
  Flower2,
} from "lucide-react";
import { request } from "../lib/api";
import type {
  Command,
  SceneDefinition,
  SceneSummary,
  WorldObject,
} from "../lib/types";

export const SCENE_CHOICES = [
  { id: "lab", label: "Laboratory", Icon: FlaskConical },
  { id: "kitchen", label: "Kitchen", Icon: CookingPot },
  { id: "living-room", label: "Living room", Icon: Sofa },
  { id: "bedroom", label: "Bedroom", Icon: BedDouble },
  { id: "garden", label: "Garden", Icon: Flower2 },
];
export function SceneChooser({
  scene,
  command,
  disabled,
}: {
  scene: SceneSummary;
  command: Command;
  disabled: boolean;
}) {
  return (
    <section className="scene-chooser" aria-label="Choose a scene">
      <div className="scene-choice-title">
        <span>ENVIRONMENTS</span>
        <small>Changing scenes resets the experiment</small>
      </div>
      <div className="scene-options">
        {SCENE_CHOICES.map(({ id, label, Icon }) => (
          <button
            key={id}
            disabled={disabled}
            aria-pressed={scene.id === id}
            className={scene.id === id ? "active" : ""}
            onClick={() => void command("scene", { scene_id: id })}
          >
            <Icon size={18} strokeWidth={1.5} />
            <span>{label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
export function useWorld(scene: SceneSummary) {
  const [definition, setDefinition] = useState<SceneDefinition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, retry] = useState(0);
  useEffect(() => {
    let active = true;
    request<SceneDefinition>(`/scenes/${scene.id}`)
      .then((result) => {
        if (result.version !== scene.version)
          throw new Error(
            "Scene changed on the server. Reload to synchronize the view.",
          );
        if (active) {
          setDefinition(result);
          setError(null);
        }
      })
      .catch((e) => {
        if (active) setError((e as Error).message);
      });
    return () => {
      active = false;
    };
  }, [scene.id, scene.version, attempt]);
  return {
    definition:
      definition?.id === scene.id && definition.version === scene.version
        ? definition
        : null,
    error,
    retry: () => {
      setError(null);
      retry((n) => n + 1);
    },
  };
}
const Solid = memo(function Solid({
  object,
  selected,
  onSelect,
}: {
  object: WorldObject;
  selected: boolean;
  onSelect: (object: WorldObject) => void;
}) {
  const { position, quaternion: q, size: s, shape, material } = object;
  return (
    <group position={position} quaternion={[q[1], q[2], q[3], q[0]]}>
      <mesh
        castShadow
        receiveShadow
        rotation={shape === "cylinder" ? [Math.PI / 2, 0, 0] : [0, 0, 0]}
        scale={shape === "ellipsoid" ? [s[0], s[1], s[2]] : [1, 1, 1]}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(object);
        }}
      >
        {shape === "box" ? (
          <boxGeometry args={[s[0] * 2, s[1] * 2, s[2] * 2]} />
        ) : shape === "cylinder" ? (
          <cylinderGeometry args={[s[0], s[0], s[1] * 2, 24]} />
        ) : (
          <sphereGeometry args={[1, 24, 16]} />
        )}
        <meshStandardMaterial
          color={object.color}
          roughness={
            material === "metal"
              ? 0.3
              : material === "tile"
                ? 0.4
                : material === "fabric"
                  ? 1
                  : 0.78
          }
          metalness={material === "metal" ? 0.65 : 0}
          emissive={
            selected
              ? "#94a759"
              : material === "luminous"
                ? object.color
                : "#000000"
          }
          emissiveIntensity={selected ? 0.25 : 0.14}
        />
      </mesh>
    </group>
  );
});
export const World = memo(function World({
  definition,
  selected,
  onSelect,
}: {
  definition: SceneDefinition;
  selected: string | null;
  onSelect: (object: WorldObject) => void;
}) {
  return (
    <group>
      <mesh position={[0, 0, -20]} receiveShadow>
        <boxGeometry args={[definition.extent[0], definition.extent[1], 40]} />
        <meshStandardMaterial color={definition.floor_color} roughness={0.9} />
      </mesh>
      {definition.objects.map((object) => (
        <Solid
          key={object.id}
          object={object}
          selected={selected === object.id}
          onSelect={onSelect}
        />
      ))}
    </group>
  );
});
