import { formatDuration } from "@/model/format";
import type { Run } from "@/model/run";

/** The toolbar pill: Live, Finished, Failed…, with the elapsed time and, while it runs, the time limit. */
export function RunStatus({ run, elapsed, limit, offline }: { run: Run; elapsed?: number; limit?: number; offline?: boolean }) {
  // Not reaching a run says nothing about how it went, so it is shown as neutral, not failed.
  const status = offline ? "idle" : run.status;
  const label = run.stopped ? "Stopped" : { idle: "Not connected", connecting: "Connecting", running: "Live", finished: "Finished", failed: "Failed" }[status];
  return (
    <div className={`run-status ${run.stopped ? "stopped" : status}`} role="status">
      <span className="dot" aria-hidden />
      <span>{label}</span>
      {elapsed !== undefined && (
        <span className="run-elapsed">
          {formatDuration(elapsed)}
          {limit && status === "running" ? <span className="run-limit"> of {formatDuration(limit)}</span> : null}
        </span>
      )}
    </div>
  );
}
