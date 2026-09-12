export type RegionId =
  "optic" | "antennal" | "mushroom" | "descending" | "vnc" | "motor";
export type Region = {
  id: RegionId;
  name: string;
  start: number;
  end: number;
  activity: number;
  stimulated: boolean;
  count?: number;
};
export type BodyPose = {
  name: string;
  parent: string | null;
  mesh: string;
  mirror: boolean;
  position: [number, number, number];
  quaternion: [number, number, number, number];
};
export type Simulation = {
  timing?: {
    simulated_seconds: number;
    compute_wall_seconds: number;
    last_step_wall_seconds: number;
    simulated_per_wall_second: number | null;
    neural_dt_ms: number | null;
  };
  scene: SceneSummary;
  episode: number;
  senses: SensoryFrame;
  time: number;
  steps: number;
  running: boolean;
  speed: number;
  odor: string;
  intensity: number;
  selected: RegionId;
  silenced: string[];
  regions: Region[];
  neurons: number[];
  approach_probability: number | null;
  environment?: {
    source: [number, number, number];
    mode: string;
    distance: number;
    concentration: number;
    antennae: number[];
  };
  body: {
    bodies: BodyPose[];
    pretarsi?: {
      name: string;
      position: [number, number, number];
      quaternion: [number, number, number, number];
      capsules: number[][];
      activation: number;
    }[];
    additional_geometry?: NonNullable<Simulation["body"]["pretarsi"]>;
    peripheral_muscles?: {
      name: string;
      target: string;
      region: string;
      side: string;
      activation: number;
      force_uN: number;
      length_mm: number;
      force_parameter_uN: number;
      joints: string[];
      interpretation: string;
    }[];
    feet: number[][];
    foot_forces: number[];
    foot_contacts: boolean[];
    joint_velocities: number[];
    muscle_activation: number[];
    muscle_force: number[];
    upright: number;
    heading: number;
    anatomy: {
      name: string;
      segments: number;
      joints: number;
      actuated_joints: number;
      muscles: number;
      mass_mg: number;
      length_unit: string;
      force_unit: string;
      specimen: string;
      muscle_map: string;
    };
    joints: number[];
    position: number[];
    velocity: number[];
    speed: number;
    activation: number[];
    force: number[];
    contacts: number;
  };
  history: { time: number; neural: number; muscle: number; speed: number }[];
  model: {
    ready?: boolean;
    device?: string;
    precision?: string;
    dynamics_version?: string;
    name: string;
    neurons: number;
    edges: number;
    engine: string;
    measured_connectome: boolean;
    controller?: "synthetic" | "posture" | "connectome";
    neural_wall_seconds?: number;
    spikes?: number;
    assumptions?: string;
  };
};
export type SensorySettings = {
  vision_model: "legacy-grid-v2" | "compound-retina-v1";
  vision_enabled: boolean;
  hearing_enabled: boolean;
  wind_enabled: boolean;
  touch_enabled: boolean;
  proprioception_enabled: boolean;
  illumination: number;
  sound_amplitude: number;
  sound_frequency: number;
  wind_speed: number;
  wind_direction: number;
  stimulus_position: [number, number, number];
};
export type SensoryFrame = {
  scene_id: string;
  time: number;
  version: string;
  settings: SensorySettings;
  retinal_routing?: {
    mapped: number;
    unresolved: number;
    registration: string;
  } | null;
  vision: {
    model: string;
    optics: string;
    eyes: {
      count: number;
      angles_degrees: number[][];
      channels: Record<string, number[]>;
    }[];
    width: number;
    height: number;
    pixels: number[][][];
    mean: number[];
    nearest_surface_mm: (number | null)[];
  };
  hearing: number[];
  wind: number[];
  gravity: number[];
  odor: number[];
  touch: number[];
  proprioception: number[];
  neural_routing: Record<string, number> | null;
};
export type Evaluation = {
  approach_A: number;
  approach_B: number;
  expected_reward: number;
  accuracy: number;
};
export type Training = {
  running: boolean;
  episodes: number;
  total: number;
  history: { episode: number; reward: number; loss: number }[];
  before: Evaluation | null;
  after: Evaluation | null;
  error: string | null;
  checkpoint: string | null;
  reward_odor: string;
  parameter_change: number;
  applied: boolean;
};
export type Command = (
  action: string,
  values?: Record<string, unknown>,
) => Promise<void>;
export type Dataset = {
  dataset: string;
  available?: boolean;
  annotation_rows: number;
  eligible_annotation_rows: number;
  source_connection_rows: number;
  edge_count: number;
  synapse_count: number;
  boundary_connection_rows: number;
  boundary_synapse_count: number;
  selection: string;
  assumptions: string;
  seed: number;
  neurons: {
    body_id: number;
    type: string;
    class: string;
    side: string;
    position: number[] | null;
    transmitter: string | null;
    transmitter_confidence: number | null;
  }[];
  sources: { file: string; url: string; bytes: number; sha256: string }[];
};
export type Probe = {
  stimulated: number;
  duration_ms: number;
  active_neurons: number;
  spikes: number;
  neurons: { body_id: number; type: string; spikes: number }[];
  history: { time_ms: number; voltage: number; population_spikes: number }[];
  physiology: string;
};

export type SceneSummary = {
  id: string;
  title: string;
  description: string;
  version: string;
  extent: [number, number, number];
  spawn: [number, number, number];
  spawn_label: string;
  floor_color: string;
  background: string;
  odor_source: [number, number, number];
  sound_source: [number, number, number];
  camera: {
    target: [number, number, number];
    position: [number, number, number];
  };
  solid_count: number;
};
export type WorldObject = {
  id: string;
  label: string;
  shape: "box" | "cylinder" | "ellipsoid";
  position: [number, number, number];
  size: number[];
  quaternion: [number, number, number, number];
  color: string;
  material: string;
  category: string;
};
export type SceneDefinition = Omit<SceneSummary, "solid_count"> & {
  objects: WorldObject[];
};

export type AnatomyGroup = {
  id: string;
  label: string;
  kind: string;
  kind_label: string;
  count: number;
  mean_hz: number;
  active_neurons: number;
  parent: string | null;
};
export type AnatomyNeuron = {
  id: string;
  body_id: number;
  type: string | null;
  superclass: string | null;
  soma_side: string | null;
  transmitter: string | null;
  hex: [number | null, number | null];
  rate_hz: number;
  voltage: number;
  voltage_unit: string;
};
export type AnatomySnapshot = {
  schema_version: number;
  group: AnatomyGroup;
  breadcrumbs: AnatomyGroup[];
  next_kind: string | null;
  view: "groups" | "neurons" | "wiring";
  items: (AnatomyGroup | AnatomyNeuron)[];
  total: number;
  offset: number;
  limit: number;
  wiring: {
    internal: { connections: number; synapses: number };
    incoming: { connections: number; synapses: number };
    outgoing: { connections: number; synapses: number };
    routes: {
      label: string;
      incoming_connections: number;
      incoming_synapses: number;
      outgoing_connections: number;
      outgoing_synapses: number;
    }[];
  } | null;
  neural_time_ms: number;
  graph_sha256: string;
  annotation_sha256: string;
  source: string;
  membership: string;
  limitations: string;
};
