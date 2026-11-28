import type { RunEntry } from "@agentomics/contract";
import { statusOf } from "@/model/run-list";

/** One dot per planned iteration: scored (blue), running (amber), failed (red), stopped (grey) or still to come. */
export function IterationDots({ run }: { run: RunEntry }) {
  const total = run.summary?.total ?? run.iterations;
  const done = run.summary?.done ?? 0;
  const status = statusOf(run);
  if (total > 12) return <span className="run-dots-count">{done}/{total}</span>;
  const metric = run.summary?.metric ?? "score";
  return (
    <span className="run-dots" aria-label={`${done} of ${total} iterations`}>
      {Array.from({ length: total }, (_, i) => {
        const value = run.summary?.history[i];
        const last = i === Math.max(done, 1) - 1;
        const [state, title] =
          value != null ? ["done", `Iteration ${i + 1}: ${metric} ${value.toFixed(3)}`]
          : status === "failed" && last ? ["failed", `Iteration ${i + 1}: failed`]
          : status === "interrupted" && i === done ? ["stopped", `Iteration ${i + 1}: stopped`]
          : i < done ? ["failed", `Iteration ${i + 1}: no score`]
          : status === "running" && i === done ? ["running", `Iteration ${i + 1}: running`]
          : ["todo", `Iteration ${i + 1}: not run`];
        return <i key={i} className={state} title={title} />;
      })}
    </span>
  );
}
