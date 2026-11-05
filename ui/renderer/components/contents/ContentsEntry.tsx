import { formatDuration } from "@/model/format";
import type { Entry } from "@/model/run";
import { StatusMark } from "../notebook/StatusMark";

/** A step in the contents: its state, code, title and how long it took; clicking jumps to it. */
/** starting: a live run is about to begin this step, so it shows as in progress. */
export function ContentsEntry({ entry, reading, starting, now, onGoTo }: { entry: Entry; reading: boolean; starting?: boolean; now: number; onGoTo: () => void }) {
  const status = starting ? "running" : entry.status;
  return (
    <button
      className={`toc-entry ${status} ${reading ? "reading" : ""}`}
      onClick={onGoTo}
      disabled={entry.status === "upcoming" || entry.status === "skipped"}
      title={entry.status === "skipped" ? "Skipped in this iteration" : undefined}
      aria-current={reading ? "true" : undefined}
    >
      <StatusMark status={status} />
      <span className="toc-code">{entry.key}</span>
      <span className="toc-title">{entry.title}</span>
      <span className="toc-time">
        {entry.startedAt ? formatDuration((entry.endedAt ?? (entry.status === "running" ? now : entry.startedAt)) - entry.startedAt) : ""}
      </span>
    </button>
  );
}
