import type { EventsSource } from "@agentomics/contract/protocol";
import { LoaderCircle } from "lucide-react";
import { formatDuration } from "@/model/format";
import type { Run } from "@/model/run";
import { Unavailable } from "../controls/Unavailable";

/** What the page says above the entries: no run open, opening, starting, can't reach the run, or the run stopped. */
export function NotebookStatus({ run, source, unreachable, now, onRetry }: {
  run: Run; source: EventsSource | null; unreachable: string | null; now: number; onRetry: (source: EventsSource) => void;
}) {
  const hasEntries = run.entries.length > 0;
  const started = run.entries.some((e) => e.position > 0 || e.records.length > 0);
  return (
    <>
      {run.status === "idle" && (
        <div className="page-message">
          <h1>No run open</h1>
          <p>Open a run from the Agentomics app.</p>
        </div>
      )}
      {run.status === "connecting" && !hasEntries && (
        <div className="page-message">
          <LoaderCircle size={18} className="spin" aria-hidden />
          <p>Opening the notebook. A run that was just started takes a moment to boot its container.</p>
        </div>
      )}
      {run.status === "running" && !started && (
        <div className="page-message" role="status">
          <LoaderCircle size={18} className="spin" aria-hidden />
          <p>
            <strong>Starting the run</strong>
            {run.startedAt && <span className="page-message-time"> · {formatDuration(Math.max(0, now - run.startedAt))}</span>}
            <br />
            The agent&apos;s container is starting and the dataset is being prepared. The first step appears here when it begins.
          </p>
        </div>
      )}
      {unreachable && (
        <Unavailable
          compact={hasEntries}
          title={hasEntries ? "Lost Connection to the Run" : "Can't Open This Run"}
          message="The run's event log couldn't be read."
          detail={unreachable}
          action={source ? () => onRetry(source) : undefined}
        />
      )}
      {run.error && !unreachable && <Unavailable compact={hasEntries} tone="failed" title="The Run Stopped" message={run.error} />}
    </>
  );
}
