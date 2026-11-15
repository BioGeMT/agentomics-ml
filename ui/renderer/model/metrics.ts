// Reading an iteration's metrics from its snapshot.
import type { Snapshot } from "./run";

// Metrics on a bounded scale (MCC can dip below 0); LOG_LOSS is unbounded and stays in the table.
export const BOUNDED = new Set(["ACC", "AUPRC", "AUROC", "F1", "MCC"]);

// The order the held-out test result lists metrics in.
export const METRIC_ORDER = ["AUROC", "AUPRC", "ACC", "F1", "MCC", "LOG_LOSS"];

export function metricNames(snapshots: Snapshot[]): string[] {
  const names = new Set<string>();
  snapshots.forEach((s) => Object.keys(s.metrics).forEach((k) => names.add(k.split("/")[1])));
  return [...names].sort();
}

export function value(snapshot: Snapshot, split: "validation" | "train", metric: string): number | undefined {
  return snapshot.metrics[`${split}/${metric}`];
}

/** Evaluation entry an iteration's metrics come from, e.g. "2.9". */
export function sourceEntry(snapshot: Snapshot): string {
  return `${snapshot.iteration + 1}.9`;
}

export function bestSnapshot(snapshots: Snapshot[]): Snapshot {
  return [...snapshots].reverse().find((s) => s.is_new_best) ?? snapshots[snapshots.length - 1];
}
