import type { RunnerOutcome } from "@agentomics/contract";
import { useEffect, useState } from "react";
import { Unavailable } from "../controls/Unavailable";

const SHOW_FILE_LABEL = navigator.platform.startsWith("Mac") ? "Show in Finder" : "Show in Explorer";

/**
 * A run that failed: what agentomics-run said, from the app's history and the runner's output, so the reason
 * (a missing key, Docker not running, a traceback…) isn't hidden behind a generic message. started: whether
 * the run recorded any steps first, which the message and its placement follow.
 */
export function RunnerFailure({ workspace, fallback, started }: { workspace: string | null; fallback: string; started: boolean }) {
  const [outcome, setOutcome] = useState<RunnerOutcome | null>(null);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (workspace) window.agentomics?.runnerOutcome(workspace).then(setOutcome, () => setOutcome(null));
  }, [workspace]);

  const title = started ? "The Run Stopped" : "The Run Couldn't Start";
  if (!outcome) return <Unavailable compact={started} tone="failed" title={title} message={fallback} />;
  const exit = outcome.exitCode === null ? "" : ` with exit code ${outcome.exitCode}`;
  const message = outcome.startError
    ? `agentomics-run couldn't be started: ${outcome.startError}`
    : started
      ? `agentomics-run ended${exit}. Its last output is below.`
      : `agentomics-run ended${exit} before the run recorded any steps.`;

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
          <pre className="runner-log" aria-label="The end of agentomics-run's output">{outcome.log}</pre>
          <div className="runner-log-actions">
            <button onClick={copy}>{copied ? "Copied" : "Copy Output"}</button>
            <button onClick={() => window.agentomics?.showFile(outcome.logFile)}>{SHOW_FILE_LABEL}</button>
          </div>
        </>
      ) : (
        <p className="runner-log-empty">agentomics-run printed nothing before it stopped.</p>
      )}
    </div>
  );
}
