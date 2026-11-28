import type { RunEntry } from "@agentomics/contract";
import { statusOf } from "@/model/run-list";

/** The run's headline number: the held-out test score once evaluated, else its best validation score. */
export function RunScore({ run }: { run: RunEntry }) {
  const value = run.summary?.test ?? run.summary?.best;
  return (
    <span className="run-score-block">
      {value == null ? (
        <span className="run-score none">{statusOf(run) === "failed" ? "Failed" : "—"}</span>
      ) : (
        <span className="run-score">{value.toFixed(3)}</span>
      )}
    </span>
  );
}
