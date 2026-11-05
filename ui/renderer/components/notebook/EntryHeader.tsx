import { formatClock, formatDuration } from "@/model/format";
import type { Entry } from "@/model/run";
import { formatCost, usageTitle } from "@/model/usage";
import { StatusMark } from "./StatusMark";

/** An entry's code and title, with when it started, its state, how long it took and what it cost. */
export function EntryHeader({ entry, now }: { entry: Entry; now: number }) {
  return (
    <header className="entry-head">
      <div className="margin"><time>{formatClock(entry.startedAt)}</time></div>
      <div className="entry-title">
        <span className="entry-code">{entry.key}</span>
        <h2>{entry.title}</h2>
        <span className={`entry-status ${entry.status}`}>
          <StatusMark status={entry.status} />
          {entry.status === "running" ? "In progress" : entry.status === "failed" ? "Failed" : "Done"}
          {entry.startedAt && ` · ${formatDuration((entry.endedAt ?? now) - entry.startedAt)}`}
          {entry.usage && <span className="entry-cost" title={usageTitle(entry.usage)}> · {formatCost(entry.usage)}</span>}
        </span>
      </div>
    </header>
  );
}
