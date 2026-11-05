import { LoaderCircle } from "lucide-react";

/** What the run will do (or what is missing), and Start Run. */
export function StartBar({ hint, disabled, starting, onStart }: { hint: string; disabled: boolean; starting: boolean; onStart: () => void }) {
  return (
    <div className="lobby-actions">
      <span className="hint">{hint}</span>
      <button className="mac-primary" disabled={disabled} onClick={onStart}>
        {starting && <LoaderCircle size={14} className="spin" aria-hidden />}
        {starting ? "Starting…" : "Start Run"}
      </button>
    </div>
  );
}
