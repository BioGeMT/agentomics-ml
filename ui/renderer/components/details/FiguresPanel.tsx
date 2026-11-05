import type { EvaluationState } from "@/hooks/useEvaluation";
import type { Run } from "@/model/run";
import { CostTable } from "../figures/CostTable";
import { Diagnostics } from "../figures/Diagnostics";
import { Figures } from "../figures/Figures";
import { TestResultCard } from "../figures/TestResultCard";
import { Timeline } from "../figures/Timeline";

/** Everything the run measured, in reading order: the test result, the metrics, the diagnostics, time and cost. */
export function FiguresPanel({ run, evaluation, provider, now, onGoTo }: {
  run: Run; evaluation: EvaluationState; provider?: string; now: number; onGoTo: (key: string) => void;
}) {
  const best = [...run.snapshots].reverse().find((s) => s.is_new_best);
  const primary = run.snapshots[run.snapshots.length - 1]?.validation_metric ?? "AUROC";
  return (
    <>
      {evaluation.test && <TestResultCard test={evaluation.test} best={best} primary={primary} />}
      <Figures snapshots={run.snapshots} onGoTo={onGoTo} />
      <Diagnostics iterations={evaluation.iterations} test={evaluation.test} best={best} names={evaluation.names} />
      <Timeline entries={run.entries} now={now} onGoTo={onGoTo} />
      <CostTable entries={run.entries} provider={provider} />
    </>
  );
}
