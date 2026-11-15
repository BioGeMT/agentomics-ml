// A run's steps and their agent traces, read as AG-UI events. Agentomics records each agentic step's
// messages in logs/traces/iteration_<n>/<step_id>.jsonl (pydantic-ai's own JSON, one message per line,
// written as they happen); a step's archived run/…/<step_id>/output.json says it is done. Steps without
// an agent (validation evaluation) have no trace, only their output.
import { randomUUID } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

export type AguiEvent = { type: string; timestamp?: number } & Record<string, unknown>;

export type Step = {
  key: string;
  iteration: number;
  stepId: string;
  /** The step's trace file, if its agent has written one. */
  trace?: string;
  /** When its first message was recorded. */
  firstAt?: number;
  /** When its output was written: the step is done. */
  doneAt?: number;
};

// The names Agentomics' steps go by (their display_name); the run page groups and orders entries by them.
const STEP_TITLES: Record<string, string> = {
  iteration_plan: "ITERATION PLAN",
  data_exploration: "DATA EXPLORATION",
  data_split: "SPLITTING",
  data_representation: "REPRESENTATION",
  model_architecture: "ARCHITECTURE",
  model_training: "TRAINING",
  model_inference: "INFERENCE",
  prediction_exploration: "PREDICTION EXPLORATION",
  validation_evaluation: "VALIDATION EVALUATION",
};

export const TRACES_DIR = path.join("logs", "traces");
const OUTPUT_FILE = "output.json";

export function stepName(step: Step): string {
  return `Iteration ${step.iteration}: ${STEP_TITLES[step.stepId] ?? step.stepId.replace(/_/g, " ").toUpperCase()}`;
}

const iterationOf = (name: string) => {
  const match = /^iteration_(\d+)$/.exec(name);
  return match ? Number(match[1]) : null;
};

const millis = (iso: unknown): number | undefined => {
  const at = typeof iso === "string" ? Date.parse(iso) : NaN;
  return Number.isNaN(at) ? undefined : at;
};

async function entries(folder: string): Promise<string[]> {
  return readdir(folder).catch(() => []);
}

async function firstMessageAt(file: string): Promise<number | undefined> {
  const text = await readFile(file, "utf8").catch(() => "");
  const line = text.slice(0, text.indexOf("\n") + 1 || undefined).trim();
  try {
    return millis(JSON.parse(line).timestamp);
  } catch {
    return undefined;
  }
}

/** The run's steps so far, in the order they ran: from its traces and its archived step outputs. */
export async function scanSteps(runFolder: string): Promise<Step[]> {
  const steps = new Map<string, Step>();
  const step = (iteration: number, stepId: string) => {
    const key = `${iteration}/${stepId}`;
    if (!steps.has(key)) steps.set(key, { key, iteration, stepId });
    return steps.get(key)!;
  };

  const traces = path.join(runFolder, TRACES_DIR);
  for (const folder of await entries(traces)) {
    const iteration = iterationOf(folder);
    if (iteration === null) continue;
    for (const file of await entries(path.join(traces, folder))) {
      if (!file.endsWith(".jsonl")) continue;
      const found = step(iteration, file.slice(0, -".jsonl".length));
      found.trace = path.join(traces, folder, file);
      found.firstAt = await firstMessageAt(found.trace);
    }
  }

  // Archived steps: run/iteration_<n>/<step_id>/, and the iteration in progress, run/current_iteration/.
  const run = path.join(runFolder, "run");
  for (const folder of await entries(run)) {
    let iteration = iterationOf(folder);
    if (folder === "current_iteration") {
      const metadata = await readFile(path.join(run, folder, "runtime_info", "iteration_metadata.json"), "utf8").catch(() => "");
      try {
        iteration = JSON.parse(metadata).iteration;
      } catch {}
    }
    if (typeof iteration !== "number") continue;
    for (const stepId of await entries(path.join(run, folder))) {
      const output = await stat(path.join(run, folder, stepId, OUTPUT_FILE)).catch(() => null);
      if (output?.isFile()) step(iteration, stepId).doneAt = output.mtimeMs;
    }
  }

  const order = (s: Step) => s.firstAt ?? s.doneAt ?? Infinity;
  return [...steps.values()].sort((a, b) => a.iteration - b.iteration || order(a) - order(b));
}

const asText = (content: unknown) => (typeof content === "string" ? content : JSON.stringify(content));

/** The AG-UI events for one pydantic-ai message from a trace. */
export function messageEvents(message: any): AguiEvent[] {
  const events: AguiEvent[] = [];
  const at = millis(message?.timestamp);
  if (message?.kind === "request") {
    for (const part of message.parts ?? []) {
      // A tool's result, or the feedback on a tool call the model got wrong.
      if ((part.part_kind === "tool-return" || part.part_kind === "retry-prompt") && part.tool_call_id) {
        const content = part.part_kind === "retry-prompt" ? `${asText(part.content)}\n\nFix the errors and try again.` : asText(part.content);
        events.push({ type: "TOOL_CALL_RESULT", messageId: randomUUID(), toolCallId: part.tool_call_id, content, role: "tool", timestamp: millis(part.timestamp) ?? at });
      }
    }
  } else if (message?.kind === "response") {
    const messageId = randomUUID();
    for (const part of message.parts ?? []) {
      if (part.part_kind === "thinking") {
        const id = randomUUID();
        events.push({ type: "REASONING_START", messageId: id, timestamp: at }, { type: "REASONING_END", messageId: id, timestamp: at });
      } else if (part.part_kind === "text" && part.content) {
        const id = randomUUID();
        events.push(
          { type: "TEXT_MESSAGE_START", messageId: id, role: "assistant", timestamp: at },
          { type: "TEXT_MESSAGE_CONTENT", messageId: id, delta: part.content, timestamp: at },
          { type: "TEXT_MESSAGE_END", messageId: id, timestamp: at },
        );
      } else if (part.part_kind === "tool-call") {
        const args = typeof part.args === "string" ? part.args : JSON.stringify(part.args ?? {});
        events.push(
          { type: "TOOL_CALL_START", toolCallId: part.tool_call_id, toolCallName: part.tool_name, parentMessageId: messageId, timestamp: at },
          { type: "TOOL_CALL_ARGS", toolCallId: part.tool_call_id, delta: args, timestamp: at },
          { type: "TOOL_CALL_END", toolCallId: part.tool_call_id, timestamp: at },
        );
      }
    }
    const usage = message.usage ?? {};
    events.push({
      type: "CUSTOM",
      name: "usage",
      value: {
        model: message.model_name ?? null,
        provider: message.provider_name ?? null,
        input_tokens: usage.input_tokens ?? 0,
        output_tokens: usage.output_tokens ?? 0,
        cache_read_tokens: usage.cache_read_tokens ?? 0,
        cost_usd: usage.cost === null || usage.cost === undefined ? null : Number(usage.cost),
      },
      timestamp: at,
    });
  }
  return events;
}
