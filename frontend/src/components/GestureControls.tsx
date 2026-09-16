import { useViewState } from "../lib/viewState";
import { useState, useEffect, useRef, useId } from "react";
import { FolderOpen, X } from "lucide-react";
import {
  GESTURES,
  type GestureController,
  type GestureId,
} from "../lib/gestures";
import "./GestureLab.css";
import { WeightPicker } from "./WeightVersions";
import Select from "./Select";

function SingleHandIcon({ gesture }: { gesture: "palm" | "fist" | "point" }) {
  const path =
    gesture === "palm"
      ? "M8 13V6a1.5 1.5 0 0 1 3 0v5-7a1.5 1.5 0 0 1 3 0v7-5a1.5 1.5 0 0 1 3 0v6-3a1.5 1.5 0 0 1 3 0v7c0 4-2 6-6 6h-1c-2 0-4-1-5-3l-4-5a1.6 1.6 0 0 1 2-2l2 1Z"
      : gesture === "fist"
        ? "M5 12V8a2 2 0 0 1 4 0V6a2 2 0 0 1 4 0v1a2 2 0 0 1 4 0v1a2 2 0 0 1 4 0v7c0 4-3 6-7 6h-2c-4 0-7-3-7-6v-1a2 2 0 0 1 2-2h7a2 2 0 0 1 0 4h-3M9 8v3m4-4v4m4-3v3"
        : "M8 13V4a2 2 0 0 1 4 0v7a2 2 0 0 1 4 0v1a2 2 0 0 1 4 0v4c0 4-3 6-7 6-3 0-5-2-7-4l-2-3a2 2 0 0 1 3-2l1 1Z";
  return (
    <svg
      viewBox="0 0 26 26"
      width="22"
      height="22"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={path} />
    </svg>
  );
}
export function GestureIcon({ gesture }: { gesture: GestureId }) {
  const left =
    gesture === "fist"
      ? "fist"
      : gesture === "point" || gesture === "point_both"
        ? "point"
        : "palm";
  const right =
    gesture === "fist"
      ? "fist"
      : gesture === "point_right" || gesture === "point_both"
        ? "point"
        : "palm";
  return (
    <span className="paired-hand-icon" aria-hidden="true">
      <span className="left-hand-icon">
        <SingleHandIcon gesture={left} />
      </span>
      <SingleHandIcon gesture={right} />
    </span>
  );
}
export function GestureControls({
  selected,
  disabled,
  onSelect,
}: {
  selected?: GestureId;
  disabled: boolean;
  onSelect: (id: GestureId | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    const place = () => {
      if (!menu.current || !trigger.current) return;
      const rect = trigger.current.getBoundingClientRect();
      const width = Math.min(260, innerWidth - 16);
      Object.assign(menu.current.style, {
        width: `${width}px`,
        maxHeight: `${innerHeight - 16}px`,
        left: `${Math.max(8, rect.left - width - 14)}px`,
        top: `${Math.max(8, Math.min(rect.top, innerHeight - menu.current.scrollHeight - 8))}px`,
      });
    };
    menu.current?.showPopover();
    place();
    document.addEventListener("pointerdown", outside);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      document.removeEventListener("pointerdown", outside);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);
  const choose = (gesture: GestureId | null) => {
    onSelect(gesture);
    setOpen(false);
    trigger.current?.focus();
  };
  return (
    <div
      className="hand-controls"
      ref={container}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          setOpen(false);
          trigger.current?.focus();
          event.stopPropagation();
        }
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null))
          setOpen(false);
      }}
    >
      <button
        ref={trigger}
        aria-label="Show hands"
        title="Show hands"
        aria-expanded={open}
        aria-controls={menuId}
        className={selected || open ? "active" : ""}
        onClick={() => setOpen(!open)}
      >
        <GestureIcon gesture={selected ?? "palm"} />
      </button>
      {open && (
        <div
          id={menuId}
          ref={menu}
          popover="manual"
          className="hand-popover"
          role="group"
          aria-label="Hand gestures"
        >
          <strong>Two-hand cues</strong>
          <p className="hand-menu-hint">
            Human left/right → same-named fly front leg. Facing the fly, the
            human’s left hand appears on its right.
          </p>
          {GESTURES.map((g) => (
            <button
              key={g.id}
              aria-label={`Show ${g.name.toLowerCase()}`}
              aria-pressed={selected === g.id}
              className={selected === g.id ? "active" : ""}
              disabled={disabled}
              onClick={() => choose(g.id)}
            >
              <GestureIcon gesture={g.id} />
              <span>
                {g.name}
                <small>{g.target}</small>
              </span>
            </button>
          ))}
          {selected && (
            <button
              aria-label="Remove hands"
              disabled={disabled}
              onClick={() => choose(null)}
            >
              <X size={16} />
              <span>Remove hands</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
export function GestureModelControls({
  controller,
  loaded,
}: {
  controller: GestureController;
  loaded?: string | null;
}) {
  const [selected, setSelected] = useViewState(
    "inference.weights",
    loaded ?? "",
  );
  useEffect(() => {
    if (loaded !== undefined) {
      setSelected(loaded ?? "");
      if (loaded)
        controller.setKind(
          loaded.startsWith("policy-") ? "transformer" : "connectome",
        );
    }
  }, [loaded]);
  const { status, busy, act } = controller;
  return (
    <div className="toolbar-model-controls">
      <Select
        aria-label="Controller family"
        value={controller.kind}
        disabled={busy || status?.running || status?.resumable}
        onChange={(value) => {
          controller.setKind(value as typeof controller.kind);
          setSelected("");
        }}
        options={[
          { value: "transformer", label: "Transformer" },
          { value: "connectome", label: "MaleCNS" },
        ]}
      />
      <WeightPicker
        versions={(status?.versions ?? []).filter(
          (v) => v.compatible !== false,
        )}
        value={selected}
        onChange={setSelected}
        disabled={busy || status?.running}
        label="Inference weights"
        baseLabel={
          controller.kind === "transformer"
            ? "New transformer · untrained"
            : "Original connectome"
        }
      />
      <button
        disabled={busy || status?.running || !status}
        onClick={() => act("/gestures/load", { checkpoint: selected || null })}
      >
        <FolderOpen size={14} />
        {busy ? "Loading…" : "Load model"}
      </button>
      <small title="Loading starts a paused lab trial using the model's recorded execution precision.">
        {loaded
          ? "Loaded: " +
            (status?.versions?.find((v) => v.id === loaded)?.name ?? loaded)
          : "Loaded: unsaved controller"}
      </small>
    </div>
  );
}
