import { formatClock } from "@/model/format";

/** The notebook's left margin: when a record happened, and an optional label under it. */
export function Margin({ at, show = true, label }: { at?: number; show?: boolean; label?: string }) {
  return (
    <div className="margin">
      {show && <time>{formatClock(at)}</time>}
      {label && <span className="margin-label">{label}</span>}
    </div>
  );
}
