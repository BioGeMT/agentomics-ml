import { useMemo, useState } from "react";
import { precisionRecall, roc, type IterationEval, type TestResult } from "@/model/evaluation";
import type { Snapshot } from "@/model/run";
import { ConfusionMatrix } from "./ConfusionMatrix";
import { CurveChart, type Series } from "./CurveChart";
import { DiagnosticsPicker } from "./DiagnosticsPicker";
import { ScoreDistribution } from "./ScoreDistribution";

/** Curves for every iteration (the chosen one highlighted) and, once evaluated, the test split. */
export function Diagnostics({ iterations, test, best, names }: { iterations: IterationEval[]; test: TestResult | null; best?: Snapshot; names: [string, string] }) {
  const bestIteration = best ? best.iteration + 1 : iterations[iterations.length - 1]?.iteration;
  const [chosen, setChosen] = useState<number | "test" | null>(null);
  const selected = chosen ?? bestIteration ?? null;
  const curves = useMemo(() => iterations.map((it) => ({ ...it, roc: roc(it.scored), pr: precisionRecall(it.scored) })), [iterations]);
  const testCurves = useMemo(() => (test?.scored ? { roc: roc(test.scored), pr: precisionRecall(test.scored) } : null), [test]);
  if (curves.length === 0) return null;
  const tone = (it: number): Series["tone"] => (it === selected ? "selected" : "other");
  const rocSeries: Series[] = [
    ...curves.map((c) => ({ key: `v${c.iteration}`, label: `Iteration ${c.iteration}`, note: `AUC ${c.roc.auc.toFixed(3)}`, points: c.roc.points, tone: tone(c.iteration) })),
    ...(testCurves ? [{ key: "test", label: "Test", note: `AUC ${testCurves.roc.auc.toFixed(3)}`, points: testCurves.roc.points, tone: "test" as const }] : []),
  ];
  const prSeries: Series[] = [
    ...curves.map((c) => ({ key: `v${c.iteration}`, label: `Iteration ${c.iteration}`, points: c.pr, tone: tone(c.iteration) })),
    ...(testCurves ? [{ key: "test", label: "Test", points: testCurves.pr, tone: "test" as const }] : []),
  ];
  const scored = selected === "test" ? test?.scored : curves.find((c) => c.iteration === selected)?.scored;
  return (
    <>
      <DiagnosticsPicker iterations={curves.map((c) => c.iteration)} hasTest={Boolean(test?.scored)} selected={selected} onSelect={setChosen} />
      <CurveChart fig="Fig. 3" title="ROC" xLabel="false positive rate" yLabel="true positive rate" series={rocSeries} diagonal />
      <CurveChart fig="Fig. 4" title="Precision–recall" xLabel="recall" yLabel="precision" series={prSeries} />
      {scored && <ConfusionMatrix scored={scored} names={names} />}
      {scored && <ScoreDistribution scored={scored} names={names} />}
    </>
  );
}
