import { formatDuration } from "@/model/format";
import type { Entry } from "@/model/run";

/**
 * What a running entry is doing right now. A command that is running already shows its own spinner,
 * so nothing is added; otherwise the model is working, shown as Thinking with how long it has been.
 */
export function LiveLine({ entry, now }: { entry: Entry; now: number }) {
  const records = entry.records.filter((r) => r.kind !== "text" || r.text.trim());
  const last = records[records.length - 1];
  if (last?.kind === "tool" && last.result === undefined) return null;
  const since =
    !last ? entry.startedAt
    : last.kind === "tool" ? last.endedAt ?? last.startedAt
    : last.kind === "text" ? last.at
    : last.startedAt;
  return (
    <div className="record record-step live-line" role="status">
      <div className="margin" />
      <div className="live-line-body">
        <span className="typing" aria-hidden><i /><i /><i /></span>
        <span className="live-line-label">Thinking</span>
        {since && <span className="live-line-time">{formatDuration(Math.max(0, now - since))}</span>}
      </div>
    </div>
  );
}
