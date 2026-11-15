import { formatDuration } from "@/model/format";
import type { Entry } from "@/model/run";

/** Between steps of a live run: which step is starting, and how long since the last one ended. */
export function NextStepLine({ entry, since, now }: { entry: Entry; since?: number; now: number }) {
  return (
    <div className="record record-step live-line" role="status">
      <div className="margin" />
      <div className="live-line-body">
        <span className="typing" aria-hidden><i /><i /><i /></span>
        <span className="live-line-label">Starting {entry.key} {entry.title}</span>
        {since && <span className="live-line-time">{formatDuration(Math.max(0, now - since))}</span>}
      </div>
    </div>
  );
}
