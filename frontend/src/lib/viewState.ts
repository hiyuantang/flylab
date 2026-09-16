import { useState, type Dispatch, type SetStateAction } from "react";

const prefix = "flylab.view.v1.";
function matchesShape(value: unknown, fallback: unknown): boolean {
  if (fallback === null) return value === null;
  if (typeof fallback === "number")
    return typeof value === "number" && Number.isFinite(value);
  if (typeof fallback !== "object") return typeof value === typeof fallback;
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) !== Array.isArray(fallback)
  )
    return false;
  return Object.entries(fallback).every(([key, expected]) =>
    matchesShape((value as Record<string, unknown>)[key], expected),
  );
}
export function readView<T>(
  key: string,
  fallback: T,
  valid?: (value: unknown) => value is T,
): T {
  try {
    const raw = localStorage.getItem(prefix + key);
    if (raw === null) return fallback;
    const value: unknown = JSON.parse(raw);
    return (valid ? valid(value) : matchesShape(value, fallback))
      ? (value as T)
      : fallback;
  } catch {
    return fallback;
  }
}
export function writeView(key: string, value: unknown) {
  try {
    localStorage.setItem(prefix + key, JSON.stringify(value));
  } catch {
    /* Storage may be disabled. */
  }
}
export function clearView(key: string) {
  try {
    localStorage.removeItem(prefix + key);
  } catch {
    /* Storage may be disabled. */
  }
}
export function useViewState<T>(
  key: string,
  fallback: T,
  valid?: (value: unknown) => value is T,
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => readView(key, fallback, valid));
  const update: Dispatch<SetStateAction<T>> = (next) =>
    setValue((previous) => {
      const result =
        typeof next === "function" ? (next as (value: T) => T)(previous) : next;
      writeView(key, result);
      return result;
    });
  return [value, update];
}
export const oneOf =
  <T extends string>(values: readonly T[]) =>
  (value: unknown): value is T =>
    typeof value === "string" && values.includes(value as T);
