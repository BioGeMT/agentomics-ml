import type { RunnerLog } from "@agentomics/contract";
import { useEffect, useRef, useState } from "react";

const POLL_MS = 2000;
const SHOW_FILE_LABEL = navigator.platform.startsWith("Mac") ? "Show in Finder" : navigator.platform.startsWith("Win") ? "Show in Explorer" : "Show in Files";
// Lines worth spotting in a long log: Python's warnings and errors, tracebacks, and the CLI's own failures.
const NOTABLE = /error|exception|traceback|failed|warning/i;

/**
 * The Output tab: what agentomics-run has printed (docker's output, warnings and errors included), followed
 * while the run goes. Kept scrolled to the end unless the reader has scrolled up to read.
 */
export function RunnerOutput({ workspace, live }: { workspace: string; live: boolean }) {
  const [output, setOutput] = useState<RunnerLog | null | undefined>(undefined);
  const [copied, setCopied] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);
  const atEnd = useRef(true);

  useEffect(() => {
    let stopped = false;
    const read = () => window.agentomics?.runnerLog(workspace).then((next) => !stopped && setOutput(next), () => !stopped && setOutput(null));
    read();
    const timer = live ? setInterval(read, POLL_MS) : undefined;
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [workspace, live]);

  useEffect(() => {
    const log = logRef.current;
    if (log && atEnd.current) log.scrollTop = log.scrollHeight;
  }, [output?.log]);

  if (output === undefined) return null;
  if (output === null) return <p className="figures-empty">This run wasn't started from the app, so its runner output isn't here.</p>;

  async function copy() {
    try {
      await navigator.clipboard.writeText(output!.log);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }

  return (
    <div className="runner-output">
      {output.log ? (
        <pre
          ref={logRef}
          className="runner-log"
          aria-label="agentomics-run's output"
          onScroll={(event) => {
            const log = event.currentTarget;
            atEnd.current = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
          }}
        >
          {output.log.split("\n").map((line, index) => (
            <span key={index} className={NOTABLE.test(line) ? "notable" : undefined}>{line}{"\n"}</span>
          ))}
        </pre>
      ) : (
        <p className="figures-empty">{live ? "Nothing printed yet." : "No runner output was captured."}</p>
      )}
      <div className="runner-log-actions">
        <button onClick={copy} disabled={!output.log}>{copied ? "Copied" : "Copy Output"}</button>
        <button onClick={() => window.agentomics?.showFile(output.logFile)}>{SHOW_FILE_LABEL}</button>
      </div>
    </div>
  );
}
