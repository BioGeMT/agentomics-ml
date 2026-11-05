// How the lobby describes runs: status names, what a run's score means, and which day it belongs to.
import type { RunEntry, RunStatus } from "@agentomics/contract";

export const STATUS_LABEL: Record<RunStatus, string> = { running: "Running", finished: "Finished", failed: "Failed", interrupted: "Stopped" };

export type RunFilter = { query: string; model: string; status: string };
export const NO_FILTER: RunFilter = { query: "", model: "", status: "" };

export function filterRuns(runs: RunEntry[], { query, model, status }: RunFilter): RunEntry[] {
  const needle = query.trim().toLowerCase();
  return runs.filter((r) =>
    (!needle || `${r.dataset} ${r.model}`.toLowerCase().includes(needle)) &&
    (!model || r.model === model) &&
    (!status || r.status === status),
  );
}

export function isFiltering({ query, model, status }: RunFilter): boolean {
  return Boolean(query.trim() || model || status);
}

/** What the score means, or how the run ended when there is none. */
export function runDetail(run: RunEntry): string {
  const s = run.summary;
  const total = s?.total ?? run.iterations;
  const current = Math.min((s?.done ?? 0) + 1, total);
  if (run.status === "running") return s?.best != null && s.metric ? `${s.metric} best so far` : `Running iteration ${current}`;
  if (run.status === "failed") return `Failed in iteration ${Math.max(s?.done ?? 1, 1)}`;
  if (s?.test != null && s.metric) return `${s.metric} on test`;
  if (s?.best != null && s.metric) return run.status === "interrupted" ? `${s.metric} on validation · stopped` : `${s.metric} on validation`;
  if (run.status === "interrupted") return `Stopped during iteration ${current}`;
  return s ? "No score recorded" : `${total} iteration${total === 1 ? "" : "s"} planned`;
}

/** "Today", "Yesterday", else the weekday and date: the run list's section headers. */
export function dayLabel(ms: number): string {
  const day = new Date(ms).toDateString();
  const today = new Date();
  const yesterday = new Date(today.getTime() - 86_400_000);
  if (day === today.toDateString()) return "Today";
  if (day === yesterday.toDateString()) return "Yesterday";
  return new Date(ms).toLocaleDateString([], { weekday: "long", day: "numeric", month: "short" });
}

/** Runs in order, grouped under the day they started. */
export function groupByDay(runs: RunEntry[]): { day: string; runs: RunEntry[] }[] {
  const groups: { day: string; runs: RunEntry[] }[] = [];
  for (const run of runs) {
    const day = dayLabel(run.startedAt);
    if (groups.at(-1)?.day === day) groups.at(-1)!.runs.push(run);
    else groups.push({ day, runs: [run] });
  }
  return groups;
}
