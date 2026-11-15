import type { RunEntry } from "@agentomics/contract";

/** One dot per planned iteration: finished, running (amber), failed (red) or still to come. */
export function IterationDots({ run }: { run: RunEntry }) {
  const total = run.summary?.total ?? run.iterations;
  const done = run.summary?.done ?? 0;
  if (total > 12) return <span className="run-dots-count">{done}/{total}</span>;
  return (
    <span className="run-dots" aria-label={`${done} of ${total} iterations`}>
      {Array.from({ length: total }, (_, i) => {
        const state =
          run.status === "failed" && i === Math.max(done, 1) - 1 ? "failed"
          : i < done ? "done"
          : run.status === "running" && i === done ? "running"
          : "todo";
        return <i key={i} className={state} />;
      })}
    </span>
  );
}
