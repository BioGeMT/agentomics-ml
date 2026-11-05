// The run history (userData/runs.json) and what the lobby shows for each run, read from its files.
import type { RunEntry, RunSummary } from "@agentomics/contract";
import { iterationFiles, iterationOfFolder, RUN_CONFIG, RUN_DIR, snapshotOf, type Snapshot } from "@agentomics/contract/snapshot";
import { app } from "electron";
import fs from "node:fs";
import path from "node:path";
import { commandOf } from "./processes";

function historyPath(): string {
  return path.join(app.getPath("userData"), "runs.json");
}

export function loadHistory(): RunEntry[] {
  try {
    return JSON.parse(fs.readFileSync(historyPath(), "utf8"));
  } catch {
    return [];
  }
}

export function saveHistory(runs: RunEntry[]): void {
  fs.mkdirSync(path.dirname(historyPath()), { recursive: true });
  fs.writeFileSync(historyPath(), JSON.stringify(runs, null, 2));
}

export function updateRun(id: string, changes: Partial<RunEntry>): void {
  const runs = loadHistory();
  const entry = runs.find((r) => r.id === id);
  if (entry) Object.assign(entry, changes);
  saveHistory(runs);
}

/**
 * Whether pid is a run's agentomics-run that is still going. It outlives the app, so a reopened app finds
 * it by its pid; after a restart that pid may belong to another program, so its command is checked too.
 */
export function isAlive(pid: number | undefined): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0); // throws for a process that is gone, or isn't the user's
  } catch {
    return false;
  }
  return commandOf(pid).includes("agentomics-run");
}

const readJson = (file: string) => {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
};

/** Each finished iteration's results, from the run's files (ui/contract/snapshot.ts). */
function snapshotsIn(workspace: string): Snapshot[] {
  const iterations = fs.readdirSync(path.join(workspace, RUN_DIR)).map(iterationOfFolder).filter((i): i is number => i !== null);
  const config = readJson(path.join(workspace, RUN_CONFIG));
  return iterations.sort((a, b) => a - b).map((iteration) => {
    const files = iterationFiles(iteration);
    return snapshotOf(iteration, config, readJson(path.join(workspace, files.state)), readJson(path.join(workspace, files.evaluation)));
  });
}

export function eventsFileOf(entry: RunEntry): string {
  return path.join(entry.workspace, "logs", "ui_events.jsonl");
}

/** "finished" or "failed" when the run's event log records its end, else null. Reads only the tail. */
export function endingOf(eventsFile: string): "finished" | "failed" | null {
  try {
    const { size } = fs.statSync(eventsFile);
    const fd = fs.openSync(eventsFile, "r");
    const length = Math.min(size, 64 * 1024);
    const tail = Buffer.alloc(length);
    fs.readSync(fd, tail, 0, length, size - length);
    fs.closeSync(fd);
    for (const line of tail.toString("utf8").split("\n").reverse()) {
      if (line.includes('"RUN_ERROR"')) return "failed";
      if (line.includes('"RUN_FINISHED"')) {
        const code = JSON.parse(line).result?.exit_code;
        return code === undefined || code === 0 ? "finished" : "failed";
      }
    }
  } catch {}
  return null;
}

const summaryCache = new Map<string, { stamp: string; summary: RunSummary }>();

/**
 * What the lobby shows for a run: its main metric at its best validation score, the held-out test score
 * once evaluated, and iterations done out of planned. Read from the run's files, cached until they change.
 */
export function runSummary(entry: RunEntry): RunSummary {
  const runDir = path.join(entry.workspace, RUN_DIR);
  const testFile = path.join(entry.workspace, "best_iteration_snapshot", "eval_predictions_test.metrics.json");
  const stamp = [runDir, testFile].map((f) => fs.statSync(f, { throwIfNoEntry: false })?.mtimeMs ?? 0).join(":");
  const cached = summaryCache.get(entry.id);
  if (cached && cached.stamp === stamp) return cached.summary;
  const summary: RunSummary = { metric: null, best: null, test: null, done: 0, total: entry.iterations ?? null, history: [] };
  try {
    for (const snapshot of snapshotsIn(entry.workspace)) {
      const metric = snapshot.validation_metric ?? "AUROC";
      const value = snapshot.metrics?.[`validation/${metric}`];
      summary.metric = metric;
      summary.done = Math.max(summary.done, (snapshot.iteration ?? 0) + 1);
      summary.total = snapshot.iterations_total ?? summary.total;
      if (typeof value === "number" && (snapshot.is_new_best || summary.best === null)) summary.best = value;
      if (typeof value === "number") summary.history[snapshot.iteration ?? summary.history.length] = value;
    }
  } catch {}
  try {
    const test = JSON.parse(fs.readFileSync(testFile, "utf8"));
    if (summary.metric && typeof test[summary.metric] === "number") summary.test = test[summary.metric];
  } catch {}
  summaryCache.set(entry.id, { stamp, summary });
  return summary;
}

/** The dataset folder the run in runFolder (a real path) was started on, so links into it can open. */
export async function datasetFolderOf(runFolder: string): Promise<string | null> {
  const entry = loadHistory().find((e) => {
    try {
      return fs.realpathSync(e.workspace) === runFolder;
    } catch {
      return false;
    }
  });
  return entry?.folder ?? null;
}
