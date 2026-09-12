import { useCallback, useEffect, useRef, useState } from "react";
import type { Simulation, Training, Command } from "./types";
export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/api${path}`,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    const data = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "The request could not be completed.",
    );
  }
  return response.json();
}
export function useWorkbench() {
  const [simulation, setSimulation] = useState<Simulation | null>(null);
  const [training, setTraining] = useState<Training | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const frames = useRef<Simulation[]>([]);
  const [replay, setReplay] = useState<number | null>(null);
  useEffect(() => {
    let disposed = false;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    let attempt = 0;
    const open = () => {
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
      );
      socket.onopen = () => {
        if (!disposed) {
          setConnected(true);
          attempt = 0;
        }
      };
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as {
          simulation: Simulation;
          training: Training;
          error: string | null;
        };
        setSimulation(data.simulation);
        setTraining(data.training);
        const last = frames.current.at(-1);
        if (last && last.time > data.simulation.time) {
          frames.current = [];
          setReplay(null);
        }
        if (!last || last.steps !== data.simulation.steps) {
          frames.current.push({ ...data.simulation, history: [] });
          if (frames.current.length > 400) frames.current.shift();
        }
        if (data.error) setError(data.error);
      };
      socket.onclose = () => {
        if (!disposed) {
          setConnected(false);
          retry = setTimeout(open, Math.min(1000 * 2 ** attempt++, 10000));
        }
      };
      socket.onerror = () => socket.close();
    };
    retry = setTimeout(open, 0);
    return () => {
      disposed = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, []);
  const command: Command = useCallback(async (action, values = {}) => {
    try {
      setError(null);
      if (action === "run" || action === "reset") setReplay(null);
      await request("/control", { action, ...values });
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  return {
    simulation,
    display:
      replay === null ? simulation : (frames.current[replay] ?? simulation),
    training,
    connected,
    error,
    setError,
    command,
    frames,
    replay,
    setReplay,
  };
}
export function downloadJSON(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
