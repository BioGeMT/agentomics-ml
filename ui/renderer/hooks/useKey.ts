import { useEffect } from "react";

/** Calls onKey when key is pressed anywhere in the window, while enabled. */
export function useKey(key: string, onKey: () => void, enabled = true): void {
  useEffect(() => {
    if (!enabled) return;
    const listener = (event: KeyboardEvent) => { if (event.key === key) onKey(); };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [key, onKey, enabled]);
}
