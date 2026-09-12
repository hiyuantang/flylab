export type NeuralLayout = {
  neurons: number;
  positioned: number;
  missing_positions: number;
  graph_sha256: string;
  coordinates: string;
  source: string;
  limits: string;
  ids: Float64Array;
  positions: Float32Array;
  vnc: Uint8Array;
};
export type NeuralItem = {
  body_id: number;
  index: number;
  type: string | null;
  superclass: string | null;
  transmitter: string | null;
  soma_side: string | null;
  position: [number, number, number] | null;
  rate_hz: number;
};
export type NeuronConnections = {
  neuron: NeuralItem & {
    voltage: number;
    voltage_unit: string;
    spike_count: number;
  };
  partners: (NeuralItem & { synapses: number })[];
  direction: "incoming" | "outgoing";
  offset: number;
  limit: number;
  total: number;
  incoming_total: number;
  outgoing_total: number;
  incoming_synapses: number;
  outgoing_synapses: number;
  neural_time_ms: number;
  graph_sha256: string;
};
