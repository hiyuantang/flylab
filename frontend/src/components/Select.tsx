import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { Check, ChevronDown } from "lucide-react";

type Option = { value: string | number; label: string; disabled?: boolean };
type Props = {
  options: Option[];
  value: string | number;
  onChange: (value: string) => void;
  "aria-label": string;
  disabled?: boolean;
  title?: string;
};

// A custom listbox in the browser's top layer also works inside modal dialogs.
// Focus stays on the combobox; navigation previews an option until committed.
export default function Select({
  options,
  value,
  onChange,
  disabled,
  title,
  "aria-label": label,
}: Props) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLSpanElement>(null);
  const search = useRef({ text: "", time: 0 });
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const selected = options.findIndex(
    (option) => String(option.value) === String(value),
  );
  const enabled = options.flatMap((option, index) =>
    option.disabled ? [] : [index],
  );

  function close() {
    panel.current?.hidePopover();
    setOpen(false);
    search.current.text = "";
  }

  function place() {
    if (!trigger.current || !panel.current) return;
    const rect = trigger.current.getBoundingClientRect();
    const margin = 8;
    const viewportWidth = document.documentElement.clientWidth;
    const below = window.innerHeight - rect.bottom - margin - 5;
    const above = rect.top - margin - 5;
    const upwards =
      below < Math.min(280, panel.current.scrollHeight) && above > below;
    const width = Math.min(
      Math.max(rect.width, 220),
      viewportWidth - margin * 2,
    );
    Object.assign(panel.current.style, {
      width: `${width}px`,
      left: `${Math.max(margin, Math.min(rect.left, viewportWidth - width - margin))}px`,
      top: upwards ? "auto" : `${rect.bottom + 5}px`,
      bottom: upwards ? `${window.innerHeight - rect.top + 5}px` : "auto",
      maxHeight: `${Math.max(0, Math.min(320, upwards ? above : below))}px`,
    });
  }

  function show(
    index = selected >= 0 && !options[selected].disabled
      ? selected
      : (enabled[0] ?? -1),
  ) {
    if (trigger.current?.matches(":disabled") || !enabled.length) return;
    setActive(index);
    panel.current?.showPopover();
    place();
    setOpen(true);
  }

  function choose(index: number) {
    const option = options[index];
    if (!option || option.disabled || trigger.current?.matches(":disabled"))
      return;
    close();
    trigger.current?.focus({ preventScroll: true });
    if (String(option.value) !== String(value)) onChange(String(option.value));
  }

  useEffect(() => {
    if (!open) return;
    if (disabled) {
      close();
      return;
    }
    const reposition = (event: Event) => {
      if (event.target instanceof Node && panel.current?.contains(event.target))
        return;
      place();
    };
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [open, disabled]);

  useEffect(() => {
    if (open)
      panel.current
        ?.querySelector(`[data-index="${active}"]`)
        ?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  function keyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const { key } = event;
    if (Date.now() - search.current.time >= 700) search.current.text = "";
    if (key === "Escape" && open) {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (key === "Tab") {
      close();
    } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(key)) {
      event.preventDefault();
      if (!open) {
        show(
          key === "End"
            ? enabled.at(-1)
            : key === "Home"
              ? enabled[0]
              : undefined,
        );
        return;
      }
      const current = enabled.indexOf(active);
      const next =
        key === "Home"
          ? 0
          : key === "End"
            ? enabled.length - 1
            : (current + (key === "ArrowDown" ? 1 : -1) + enabled.length) %
              enabled.length;
      setActive(enabled[next] ?? -1);
    } else if (key === "Enter" || (key === " " && !search.current.text)) {
      event.preventDefault();
      if (open) choose(active);
      else show();
    } else if (
      key.length === 1 &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault();
      const now = Date.now();
      const previous =
        now - search.current.time < 700 ? search.current.text : "";
      const text = previous + key.toLocaleLowerCase();
      search.current = { text, time: now };
      const query = [...text].every((character) => character === text[0])
        ? text[0]
        : text;
      const start = enabled.indexOf(open ? active : selected);
      const ordered = [
        ...enabled.slice(start + 1),
        ...enabled.slice(0, start + 1),
      ];
      const match = ordered.find((index) =>
        options[index].label.toLocaleLowerCase().startsWith(query),
      );
      if (!open) show(match);
      else if (match !== undefined) setActive(match);
    }
  }

  return (
    <span className="custom-select">
      <button
        ref={trigger}
        type="button"
        className="custom-select-trigger"
        role="combobox"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={id}
        aria-activedescendant={
          open && active >= 0 ? `${id}-${active}` : undefined
        }
        disabled={disabled || !enabled.length}
        title={title}
        onKeyDown={keyDown}
        onClick={() => (open ? close() : show())}
        onBlur={(event) => {
          if (!panel.current?.contains(event.relatedTarget)) close();
        }}
      >
        <span className="custom-select-value">
          {options[selected]?.label ?? "Select an option"}
        </span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      <span
        ref={panel}
        id={id}
        popover="auto"
        role="listbox"
        aria-label={label}
        className="custom-select-menu"
        onToggle={(event) => setOpen(event.newState === "open")}
        onMouseDown={(event) => event.preventDefault()}
      >
        {options.map((option, index) => (
          <span
            key={option.value}
            id={`${id}-${index}`}
            role="option"
            aria-selected={index === selected}
            aria-disabled={option.disabled || undefined}
            data-index={index}
            data-active={index === active}
            className="custom-select-option"
            onPointerMove={() => !option.disabled && setActive(index)}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              choose(index);
            }}
          >
            <span>{option.label}</span>
            {index === selected && <Check size={14} aria-hidden="true" />}
          </span>
        ))}
      </span>
    </span>
  );
}
