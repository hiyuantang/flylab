import { useState, type InputHTMLAttributes } from "react";

type Props = Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "type"> & {
  value: number | string;
};

// Keep incomplete edits local; never turn an erased value into zero.
// On blur, an unfinished edit returns to the last committed number.
export default function NumberInput({
  value,
  onChange,
  onBlur,
  ...props
}: Props) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      {...props}
      type="number"
      value={draft ?? value}
      onChange={(event) => {
        setDraft(event.target.value);
        if (
          event.target.value !== "" &&
          Number.isFinite(event.target.valueAsNumber)
        ) {
          onChange?.(event);
        }
      }}
      onBlur={(event) => {
        setDraft(null);
        onBlur?.(event);
      }}
    />
  );
}
