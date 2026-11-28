import { useState } from "react";
import { bestSnapshot } from "@/model/metrics";
import type { Snapshot } from "@/model/run";
import { MetricByIteration } from "./MetricByIteration";
import { MetricsTable } from "./MetricsTable";
import { TrainVsValidation } from "./TrainVsValidation";

export type GoTo = (entryKey: string) => void;

/** The metric figures: the chosen metric across iterations, train against validation, and the metrics table. */
export function Figures({ snapshots: all, live, onGoTo }: { snapshots: Snapshot[]; live: boolean; onGoTo: GoTo }) {
  // An iteration that failed before its evaluation has a folder but no metrics: nothing to plot.
  const snapshots = all.filter((s) => Object.keys(s.metrics).length > 0);
  const primary = snapshots[snapshots.length - 1]?.validation_metric ?? "AUROC";
  const [metric, setMetric] = useState<string | null>(null);
  if (snapshots.length === 0) {
    return (
      <div className="figures-empty">
        <p>{live ? "Figures appear when the first iteration finishes and its model is evaluated." : "No iteration was evaluated, so there is nothing to plot."}</p>
      </div>
    );
  }
  const shown = metric ?? primary;
  return (
    <>
      <MetricByIteration snapshots={snapshots} metric={shown} primary={primary} onGoTo={onGoTo} />
      <TrainVsValidation snapshot={bestSnapshot(snapshots)} primary={primary} onGoTo={onGoTo} />
      <MetricsTable snapshots={snapshots} primary={primary} selected={shown} onSelect={setMetric} onGoTo={onGoTo} />
    </>
  );
}
