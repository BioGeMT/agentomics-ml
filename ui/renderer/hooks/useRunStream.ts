import { HttpAgent } from "@ag-ui/client";
import { eventsUrl, type EventsSource } from "@agentomics/contract/protocol";
import { useEffect, useRef, useState } from "react";
import { applyEvent, EMPTY_RUN, type Run, type Snapshot } from "@/model/run";

const STREAM_ENDED = Symbol("stream ended");

/** The log ended without recording the run's end: the run was stopped, or logged nothing at all. */
function endStream(run: Run) {
  if (run.status !== "running" && run.status !== "connecting") return;
  run.error = run.status === "running" ? "The run stopped before it finished." : "This run didn't record its steps.";
  run.status = "failed";
  run.entries.forEach((e) => {
    if (e.status === "running") e.status = "failed";
  });
}

/**
 * A run's notebook, built from its AG-UI event stream (POST /api/events). connect() opens a run's event
 * log; the stream follows it while the run is going. A stream that ends before the run does means the
 * run was stopped.
 */
export function useRunStream() {
  const [run, setRun] = useState<Run>(EMPTY_RUN);
  const [source, setSource] = useState<EventsSource | null>(null);
  const [unreachable, setUnreachable] = useState<string | null>(null);
  const agentRef = useRef<HttpAgent | null>(null);
  const pending = useRef<any[]>([]);

  /**
   * Events arrive in bursts (a finished run's log sends hundreds at once); apply them about once a frame.
   * A timer rather than requestAnimationFrame, which stops while the window is hidden.
   */
  function apply(event: any) {
    pending.current.push(event);
    if (pending.current.length > 1) return;
    setTimeout(() => {
      const batch = pending.current;
      pending.current = [];
      setRun((current) => {
        const draft: Run = structuredClone(current);
        batch.forEach((e) => (e === STREAM_ENDED ? endStream(draft) : applyEvent(draft, e)));
        return draft;
      });
    }, 16);
  }

  function connect(next: EventsSource) {
    agentRef.current?.abortRun();
    setSource(next);
    setUnreachable(null);
    setRun({ ...EMPTY_RUN, status: "connecting" });
    const agent = new HttpAgent({ url: eventsUrl(next) });
    agentRef.current = agent;
    agent
      .runAgent({}, {
        onEvent: ({ event }) => apply(event),
      })
      .then(() => agentRef.current === agent && apply(STREAM_ENDED))
      .catch((error: any) => {
        setRun((current) => (current.status === "finished" ? current : { ...current, status: "failed" }));
        setUnreachable(String(error?.message ?? error));
      });
  }

  /** The run's finished iterations, which come from its workspace rather than its events. */
  function setSnapshots(snapshots: Snapshot[]) {
    setRun((current) => ({ ...current, snapshots }));
  }

  useEffect(() => () => agentRef.current?.abortRun(), []);

  return { run, source, unreachable, connect, setSnapshots };
}
