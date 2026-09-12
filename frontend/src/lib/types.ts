export type RegionId =
  "optic" | "antennal" | "mushroom" | "descending" | "vnc" | "motor";
export type Region = {
  id: RegionId;
  name: string;
  start: number;
  end: number;
  activity: number;
  stimulated: boolean;
};
export type BodyPose = {
  name: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
};
export type Simulation = {
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
  approach_probability: number;
  body: {
    bodies: BodyPose[];
    feet: number[][];
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
    name: string;
    neurons: number;
    edges: number;
    engine: string;
    measured_connectome: boolean;
  };
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
