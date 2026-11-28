import type { RunnerOutcome } from "@agentomics/contract";
import { useEffect, useRef, useState } from "react";
import { Unavailable } from "../controls/Unavailable";

/** Exit codes that say how a run was ended, rather than what went wrong in it. */
const EXIT_NOTES: Record<number, string> = {
  137: " Exit code 137 means its container was stopped or ran out of memory.",
  [-1]: " On Windows, exit code -1 means the process was ended from outside: by Stop, by closing its window, or by the system.",
};

const SHOW_FILE_LABEL = navigator.platform.startsWith("Mac") ? "Show in Finder" : "Show in Explorer";

/**
 * A run that failed: what agentomics-run said, from the app's history and the runner's output, so the reason
 * (a missing key, Docker not running, a traceback…) isn't hidden behind a generic message. started: whether
 * the run recorded any steps first, which the message and its placement follow.
 */
export function RunnerFailure({ workspace, fallback, started }: { workspace: string | null; fallback: string; started: boolean }) {
  const [outcome, setOutcome] = useState<RunnerOutcome | null>(null);
  const [copied, setCopied] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (workspace) window.agentomics?.runnerOutcome(workspace).then(setOutcome, () => setOutcome(null));
  }, [workspace]);
  // The reason a run failed is at the end of its output: open the log there.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [outcome?.log]);

  const title = outcome?.stopped ? "Stopped" : started ? "The Run Stopped" : "The Run Couldn't Start";
  if (!outcome) return <Unavailable compact={started} tone="failed" title={title} message={fallback} />;
  const exit = outcome.exitCode === null ? "" : ` with exit code ${outcome.exitCode}`;
  const message = outcome.startError
    ? `agentomics-run couldn't be started: ${outcome.startError}`
    : outcome.stopped
      ? `You stopped the run.${outcome.log ? " Its last output is below." : ""}`
      : `agentomics-run ended${exit}${started ? "" : " before the run recorded any steps"}.${outcome.log ? " Its last output is below." : ""}${EXIT_NOTES[outcome.exitCode ?? 0] ?? ""}`;

  async function copy() {
    try {
      await navigator.clipboard.writeText(outcome!.log);
      setCopied(true);
    } catch {}
  }

  return (
    <div className={`runner-failure ${started ? "compact" : ""}`}>
      <Unavailable compact={started} tone="failed" title={title} message={message} />
      {outcome.log ? (
        <>
          <pre className="runner-log" ref={logRef} aria-label="The end of agentomics-run's output">{outcome.log}</pre>
          <div className="runner-log-actions">
            <button onClick={copy}>{copied ? "Copied" : "Copy Output"}</button>
            <button onClick={() => window.agentomics?.showFile(outcome.logFile)}>{SHOW_FILE_LABEL}</button>
          </div>
        </>
      ) : (
        <p className="runner-log-empty">No runner output was captured.</p>
      )}
    </div>
  );
}
