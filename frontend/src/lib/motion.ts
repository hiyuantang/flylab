import { Matrix4, Quaternion, Vector3 } from "three";
import type { Simulation } from "./types";

type Transform = {
  position: Vector3;
  quaternion: Quaternion;
  parent: string | null;
};
type Frame = { time: number; nodes: Map<string, Transform> };
type WorldTransform = Transform & { matrix: Matrix4; generation: number };
const UNIT = new Vector3(1, 1, 1);

function frameFrom(sim: Simulation): Frame {
  const nodes = new Map<string, Transform>();
  for (const pose of [
    ...sim.body.bodies,
    ...(sim.body.pretarsi ?? []),
    ...(sim.body.additional_geometry ?? []),
  ]) {
    const [w, x, y, z] = pose.quaternion;
    nodes.set(pose.name, {
      position: new Vector3(...pose.position),
      quaternion: new Quaternion(x, y, z, w).normalize(),
      parent: pose.parent ?? null,
    });
  }
  sim.body.feet.forEach((foot, i) =>
    nodes.set(`foot:${i}`, {
      position: new Vector3(...foot),
      quaternion: new Quaternion(),
      parent: sim.body.foot_parents?.[i] ?? null,
    }),
  );
  // Work from immutable world transforms before converting every child to its
  // parent's frame. Interpolating world positions would detach rotating limbs.
  const matrices = new Map(
    [...nodes].map(([name, p]) => [
      name,
      new Matrix4().compose(p.position, p.quaternion, UNIT),
    ]),
  );
  for (const [name, pose] of nodes) {
    const parent = pose.parent ? matrices.get(pose.parent) : undefined;
    if (parent) {
      parent
        .clone()
        .invert()
        .multiply(matrices.get(name)!)
        .decompose(pose.position, pose.quaternion, new Vector3());
    } else pose.parent = null;
  }
  return { time: sim.time, nodes };
}

/** Buffered display only. Never predicts state or feeds interpolated poses to physics. */
export class PosePlayback {
  private frames: Frame[] = [];
  private world = new Map<string, WorldTransform>();
  private key = "";
  private arrival = 0;
  private observedRate = 0;
  private interval = 1 / 60;
  private running = false;
  private pausedRate = 0;
  private generation = 0;
  time = 0;
  get latestTime() {
    return this.frames.at(-1)?.time ?? this.time;
  }
  get bufferedFrames() {
    return this.frames.length;
  }
  get(name: string) {
    return this.world.get(name);
  }

  push(sim: Simulation, wallSeconds: number) {
    const key = `${sim.episode}:${sim.scene.id}:${sim.model.controller}:${sim.timing?.muscle_command_hz}`;
    const last = this.frames.at(-1);
    if (
      !last ||
      key !== this.key ||
      sim.time < last.time - 1e-9 ||
      sim.time - last.time > 0.5
    ) {
      this.key = key;
      this.frames = [frameFrom(sim)];
      this.world.clear();
      this.time = sim.time;
      this.arrival = wallSeconds;
      this.observedRate = 0;
      this.running = sim.running;
      this.renderAt(this.time);
      return;
    }
    if (sim.time > last.time + 1e-9) {
      const interval = sim.time - last.time;
      const measured = interval / Math.max(0.001, wallSeconds - this.arrival);
      this.observedRate = this.observedRate
        ? 0.8 * this.observedRate + 0.2 * measured
        : measured;
      this.interval = interval;
      this.arrival = wallSeconds;
      this.frames.push(frameFrom(sim));
      // Bound memory when tabs are suspended; never extrapolate beyond a sample.
      if (this.frames.length > 120) {
        this.frames.shift();
        this.time = Math.max(this.time, this.frames[0].time);
      }
      if (!sim.running) this.pausedRate = (sim.time - this.time) / 0.15;
    }
    if (this.running && !sim.running)
      this.pausedRate = (this.latestTime - this.time) / 0.15;
    this.running = sim.running;
  }

  advance(wallDelta: number) {
    const lead = this.latestTime - this.time;
    let rate = this.pausedRate;
    if (this.running) {
      // Hold roughly one snapshot of slack and adapt to actual throughput, not
      // assumed real time. A second endpoint is needed for interpolation.
      const base = this.observedRate;
      const correction =
        (lead - this.interval * 1.5) /
        Math.max(0.05, this.interval / Math.max(base, 0.001));
      rate = Math.max(0, Math.min(base * 1.3, base + correction * 0.35));
    }
    this.time = Math.min(
      this.latestTime,
      this.time + Math.max(0, Math.min(wallDelta, 0.1)) * rate,
    );
    this.renderAt(this.time);
    while (this.frames.length > 2 && this.frames[1].time <= this.time)
      this.frames.shift();
  }

  renderAt(time: number) {
    if (!this.frames.length) return;
    let right = this.frames.findIndex((frame) => frame.time >= time);
    if (right < 0) right = this.frames.length - 1;
    const b = this.frames[right],
      a = this.frames[Math.max(0, right - 1)];
    const alpha =
      b.time === a.time
        ? 1
        : Math.max(0, Math.min(1, (time - a.time) / (b.time - a.time)));
    const generation = ++this.generation;
    const resolve = (name: string): WorldTransform | undefined => {
      const end = b.nodes.get(name);
      if (!end) return;
      let out = this.world.get(name);
      if (out?.generation === generation) return out;
      if (!out) {
        out = {
          position: new Vector3(),
          quaternion: new Quaternion(),
          parent: end.parent,
          matrix: new Matrix4(),
          generation: 0,
        };
        this.world.set(name, out);
      }
      out.generation = generation;
      const start = a.nodes.get(name) ?? end;
      out.position.lerpVectors(start.position, end.position, alpha);
      out.quaternion.slerpQuaternions(start.quaternion, end.quaternion, alpha);
      out.matrix.compose(out.position, out.quaternion, UNIT);
      const parent = end.parent ? resolve(end.parent) : undefined;
      if (parent) {
        out.matrix.premultiply(parent.matrix);
        out.position.setFromMatrixPosition(out.matrix);
        out.quaternion.setFromRotationMatrix(out.matrix);
      }
      return out;
    };
    for (const name of b.nodes.keys()) resolve(name);
  }
}
