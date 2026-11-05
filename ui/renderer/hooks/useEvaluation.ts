import { useEffect, useState } from "react";
import { loadClassNames, loadIterations, loadTest, type IterationEval, type TestResult } from "@/model/evaluation";
import type { Snapshot } from "@/model/run";

export type EvaluationState = { iterations: IterationEval[]; test: TestResult | null; names: [string, string] };

/** The run's prediction files, loaded as iterations are evaluated: validation scores per iteration, the test result once finished, and the class names. */
export function useEvaluation(root: string | null, snapshots: Snapshot[], finished: boolean): EvaluationState {
  const [iterations, setIterations] = useState<IterationEval[]>([]);
  const [test, setTest] = useState<TestResult | null>(null);
  const [names, setNames] = useState<[string, string]>(["negative", "positive"]);
  useEffect(() => { if (root) loadClassNames(root).then(setNames); }, [root]);
  useEffect(() => {
    if (!root || snapshots.length === 0) return;
    let cancelled = false;
    loadIterations(root, snapshots.length).then((result) => { if (!cancelled) setIterations(result); });
    return () => { cancelled = true; };
  }, [root, snapshots.length]);
  useEffect(() => {
    if (!root || !finished) return;
    let cancelled = false;
    loadTest(root).then((result) => { if (!cancelled) setTest(result); });
    return () => { cancelled = true; };
  }, [root, finished]);
  return { iterations, test, names };
}
