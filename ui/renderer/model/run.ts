// Turns a run's AG-UI event stream into the notebook the viewer shows.
import { parseUsage, USAGE_EVENT } from "@agentomics/contract/events";
import type { Snapshot } from "@agentomics/contract/snapshot";

export type { Snapshot };

export const STEP_SEQUENCE = [
  "ITERATION PLAN",
  "DATA EXPLORATION",
  "SPLITTING",
  "REPRESENTATION",
  "ARCHITECTURE",
  "TRAINING",
  "INFERENCE",
  "PREDICTION EXPLORATION",
  "VALIDATION EVALUATION",
];

export type ToolRecord = {
  kind: "tool";
  id: string;
  name: string;
  args: string;
  result?: string;
  startedAt?: number;
  endedAt?: number;
};
export type TextRecord = { kind: "text"; id: string; text: string; at?: number };
export type ThinkingRecord = { kind: "thinking"; id: string; startedAt?: number; endedAt?: number };
export type EntryRecord = ToolRecord | TextRecord | ThinkingRecord;

export type EntryStatus = "upcoming" | "running" | "done" | "failed" | "skipped";

export type Entry = {
  key: string; // e.g. "1.4"
  iteration: number; // 1-based
  position: number; // 1-based within the iteration
  title: string; // "Data exploration"
  status: EntryStatus;
  startedAt?: number;
  endedAt?: number;
  records: EntryRecord[];
  usage?: Usage;
};

/** Model usage summed over responses. `cost` sums only priced responses; `unpriced` counts the rest. */
export type Usage = { inputTokens: number; outputTokens: number; cachedTokens: number; cost: number; responses: number; unpriced: number; provider?: string };

export function addUsage(total: Usage | undefined, more: Usage | undefined): Usage | undefined {
  if (!more) return total;
  if (!total) return { ...more };
  return {
    inputTokens: total.inputTokens + more.inputTokens,
    outputTokens: total.outputTokens + more.outputTokens,
    cachedTokens: total.cachedTokens + more.cachedTokens,
    cost: total.cost + more.cost,
    responses: total.responses + more.responses,
    unpriced: total.unpriced + more.unpriced,
    provider: total.provider ?? more.provider,
  };
}

export type RunStatus = "idle" | "connecting" | "running" | "finished" | "failed";

export type Run = {
  status: RunStatus;
  error?: string;
  startedAt?: number;
  endedAt?: number;
  entries: Entry[];
  snapshots: Snapshot[];
  lastAt?: number; // latest event time seen; stands in for events that carry none
};

export const EMPTY_RUN: Run = { status: "idle", entries: [], snapshots: [] };

export function parseStepName(stepName: string): { iteration: number; title: string; position: number } {
  const match = /^Iteration (\d+): (.+)$/.exec(stepName);
  const raw = (match ? match[2] : stepName).trim();
  const upper = raw.toUpperCase();
  const index = STEP_SEQUENCE.indexOf(upper);
  return {
    iteration: match ? Number(match[1]) + 1 : 1,
    title: upper.charAt(0) + upper.slice(1).toLowerCase(),
    position: index >= 0 ? index + 1 : 0,
  };
}

function currentEntry(run: Run): Entry {
  const running = [...run.entries].reverse().find((e) => e.status === "running");
  if (running) return running;
  if (run.entries.length === 0) {
    run.entries.push({ key: "0.0", iteration: 1, position: 0, title: "Setup", status: "running", records: [] });
  }
  return run.entries[run.entries.length - 1];
}

function findTool(run: Run, id: string): ToolRecord | undefined {
  for (let i = run.entries.length - 1; i >= 0; i--) {
    const record = run.entries[i].records.find((r) => r.kind === "tool" && r.id === id);
    if (record) return record as ToolRecord;
  }
}

/** Applies one AG-UI event to the run, in place (the caller works on a copy). */
export function applyEvent(run: Run, event: any): void {
  if (typeof event.timestamp === "number") run.lastAt = event.timestamp;
  const at: number | undefined = event.timestamp ?? run.lastAt;
  run.startedAt ??= at;
  switch (event.type) {
    case "RUN_STARTED":
      run.status = "running";
      run.startedAt ??= at;
      break;
    case "STEP_STARTED": {
      const { iteration, title, position } = parseStepName(event.stepName);
      run.entries.push({
        key: `${iteration}.${position || run.entries.filter((e) => e.iteration === iteration).length + 1}`,
        iteration,
        position,
        title,
        status: "running",
        startedAt: at,
        records: [],
      });
      break;
    }
    case "STEP_FINISHED": {
      const { iteration, position } = parseStepName(event.stepName);
      const entry = [...run.entries].reverse().find((e) => e.iteration === iteration && e.position === position && e.status === "running");
      if (entry) {
        entry.status = "done";
        entry.endedAt = at;
      }
      break;
    }
    case "TEXT_MESSAGE_START":
      currentEntry(run).records.push({ kind: "text", id: event.messageId, text: "", at });
      break;
    case "TEXT_MESSAGE_CONTENT": {
      const record = currentEntry(run).records.find((r) => r.kind === "text" && r.id === event.messageId);
      if (record && record.kind === "text") record.text += event.delta;
      break;
    }
    case "REASONING_START":
      currentEntry(run).records.push({ kind: "thinking", id: event.messageId, startedAt: at });
      break;
    case "REASONING_END": {
      const record = currentEntry(run).records.find((r) => r.kind === "thinking" && r.id === event.messageId);
      if (record && record.kind === "thinking") record.endedAt = at;
      break;
    }
    case "TOOL_CALL_START":
      currentEntry(run).records.push({ kind: "tool", id: event.toolCallId, name: event.toolCallName, args: "", startedAt: at });
      break;
    case "TOOL_CALL_ARGS": {
      const tool = findTool(run, event.toolCallId);
      if (tool) tool.args += event.delta;
      break;
    }
    case "TOOL_CALL_RESULT": {
      const tool = findTool(run, event.toolCallId);
      if (tool) {
        tool.result = event.content;
        tool.endedAt = at;
      }
      break;
    }
    case "CUSTOM": {
      const v = event.name === USAGE_EVENT ? parseUsage(event.value) : null;
      if (!v) break; // not ours, or not matching its schema
      const entry = currentEntry(run);
      entry.usage = addUsage(entry.usage, {
        inputTokens: v.input_tokens,
        outputTokens: v.output_tokens,
        cachedTokens: v.cache_read_tokens,
        cost: v.cost_usd ?? 0,
        responses: 1,
        unpriced: v.cost_usd === null ? 1 : 0,
        provider: v.provider ?? undefined,
      });
      break;
    }
    case "RUN_FINISHED": {
      // Agentomics records its exit code; a run that ends with one other than 0 failed.
      const code = event.result?.exit_code;
      run.status = code === undefined || code === 0 ? "finished" : "failed";
      if (run.status === "failed") run.error = `The run ended with exit code ${code}.`;
      run.endedAt = at;
      run.entries.forEach((e) => {
        if (e.status === "running") e.status = "done";
      });
      break;
    }
    case "RUN_ERROR":
      run.status = "failed";
      run.error = event.message;
      run.endedAt = at;
      run.entries.forEach((e) => {
        if (e.status === "running") e.status = "failed";
      });
      break;
  }
}

/** Steps an iteration passed over without running (e.g. splitting when the split is reused). */
function skipped(run: Run): Entry[] {
  const result: Entry[] = [];
  const iterations = [...new Set(run.entries.map((e) => e.iteration))];
  for (const iteration of iterations) {
    const reached = run.entries.filter((e) => e.iteration === iteration && e.position > 0);
    const furthest = Math.max(0, ...reached.map((e) => e.position));
    const finished = run.status === "finished" || run.status === "failed" || iterations.some((i) => i > iteration);
    STEP_SEQUENCE.forEach((name, index) => {
      const position = index + 1;
      if ((position < furthest || (finished && position > furthest)) && !reached.some((e) => e.position === position)) {
        result.push({ ...upcoming(iteration, position, name), status: "skipped" });
      }
    });
  }
  return result;
}

/** Entries plus skipped steps, the steps still to come in the current iteration, and chapters for later ones. */
export function withUpcoming(run: Run, totalIterations?: number): Entry[] {
  const entries = [...run.entries, ...skipped(run)];
  if (run.status === "finished" || run.status === "failed") return sortEntries(entries);
  const last = run.entries[run.entries.length - 1];
  const iteration = last?.iteration ?? 1;
  const reached = new Set(entries.filter((e) => e.iteration === iteration).map((e) => e.position));
  const fromPosition = last && last.iteration === iteration ? last.position : 0;
  STEP_SEQUENCE.forEach((name, index) => {
    const position = index + 1;
    if (position > fromPosition && !reached.has(position)) {
      entries.push(upcoming(iteration, position, name));
    }
  });
  for (let next = iteration + 1; totalIterations && next <= totalIterations; next++) {
    STEP_SEQUENCE.forEach((name, index) => entries.push(upcoming(next, index + 1, name)));
  }
  return sortEntries(entries);
}

function sortEntries(entries: Entry[]): Entry[] {
  return entries.sort((a, b) => a.iteration - b.iteration || a.position - b.position);
}

function upcoming(iteration: number, position: number, name: string): Entry {
  return {
    key: `${iteration}.${position}`,
    iteration,
    position,
    title: name.charAt(0) + name.slice(1).toLowerCase(),
    status: "upcoming",
    records: [],
  };
}

/** While a live run is between steps, or before its first, the step that comes next; otherwise undefined. */
export function nextStep(run: Run, entries: Entry[]): Entry | undefined {
  if (run.status !== "running" || run.entries.some((e) => e.status === "running" && e.position > 0)) return undefined;
  return entries.find((e) => e.status === "upcoming");
}
