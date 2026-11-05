// The run page's own address. The lobby builds it; the run page reads it.
//
//   /run/?log=<run>/logs/ui_events.jsonl   a run, read from its event log
//   ...&live=1                             the run is going: keep its newest step in view
//   ...&dataset=&model=&iterations=&provider=   what to title the page with before events arrive
import type { RunEntry } from "@agentomics/contract";
import type { EventsSource } from "@agentomics/contract/protocol";

export type RunMeta = { dataset?: string; model?: string; iterations?: number; provider?: string };

export type RunLink = {
  source: EventsSource;
  /** The run's folder on disk, for the file explorer and reports: two levels above its event log. */
  workspace: string;
  live: boolean;
  meta: RunMeta;
};

const LOG_IN_WORKSPACE = "logs/ui_events.jsonl";

export function runLink(run: RunEntry): string {
  const query = new URLSearchParams({ log: run.eventsFile ?? `${run.workspace}/${LOG_IN_WORKSPACE}` });
  if (run.status === "running") query.set("live", "1");
  query.set("dataset", run.dataset);
  query.set("model", run.model);
  query.set("iterations", String(run.iterations));
  if (run.provider) query.set("provider", run.provider);
  return `/run/?${query}`;
}

/** The link's run, or null when it names no event log. */
export function parseRunLink(search: string): RunLink | null {
  const query = new URLSearchParams(search);
  const log = query.get("log");
  if (!log) return null;
  const meta: RunMeta = {
    dataset: query.get("dataset") ?? undefined,
    model: query.get("model") ?? undefined,
    iterations: query.get("iterations") ? Number(query.get("iterations")) : undefined,
    provider: query.get("provider") ?? undefined,
  };
  return { source: { log }, workspace: log.split("/").slice(0, -2).join("/"), live: query.get("live") === "1", meta };
}
