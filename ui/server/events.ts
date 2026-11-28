// POST /api/events?run=<run folder>: a run's AG-UI events as text/event-stream (see ui/contract/protocol.ts).
// Made from the run's agent traces as they grow (./traces.ts). Runs recorded before traces existed have a
// ready-made AG-UI log, logs/ui_events.jsonl, which is replayed as it is.
import { parseEventsQuery } from "@agentomics/contract/protocol";
import { open, stat } from "node:fs/promises";
import path from "node:path";
import type { Roots } from "./roots";
import { messageEvents, scanSteps, stepName, TRACES_DIR, type AguiEvent, type Step } from "./traces";

const SSE_HEADERS = { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" };
const LEGACY_LOG = path.join("logs", "ui_events.jsonl");
const POLL_MS = 500;

/** Whether a run is still going, from the app's run history; its traces may still grow while it is. */
export type IsRunning = (runFolder: string) => Promise<boolean>;
/** How a finished run ended, from the app's run history: its exit code, or why it never got one. Null if unknown. */
export type RunEnding = (runFolder: string) => Promise<{ exitCode: number | null; error?: string; endedAt?: number; stopped?: boolean } | null>;

const sleep = () => new Promise((resolve) => setTimeout(resolve, POLL_MS));

/** Reads the complete lines appended to a file since the last read. Only files inside the runs folders. */
function lineReader(roots: Roots, file: string) {
  let position = 0;
  let decoder = new TextDecoder();
  let pending = "";
  return async function readLines(): Promise<string[]> {
    if (!(await roots.within(file))) return []; // not there yet, or a link out of the runs folders
    const size = await stat(file).then((s) => s.size, () => null);
    if (size === null || size === position) return [];
    if (size < position) [position, pending, decoder] = [0, "", new TextDecoder()]; // written afresh
    const handle = await open(file);
    try {
      const { buffer, bytesRead } = await handle.read(Buffer.alloc(size - position), 0, size - position, position);
      position += bytesRead;
      pending += decoder.decode(buffer.subarray(0, bytesRead), { stream: true });
    } finally {
      await handle.close();
    }
    const lines = pending.split("\n");
    pending = lines.pop() ?? ""; // a line still being written
    return lines.filter((line) => line.trim());
  };
}

export function createEvents(roots: Roots, isRunning: IsRunning = async () => false, runEnding: RunEnding = async () => null) {
  function stream(produce: (send: (event: AguiEvent | string) => void, signal: AbortSignal | undefined) => Promise<void>, signal?: AbortSignal) {
    const encoder = new TextEncoder();
    return new Response(
      new ReadableStream({
        async start(controller) {
          const send = (event: AguiEvent | string) => {
            // AG-UI timestamps are whole milliseconds; file times have fractions.
            if (typeof event !== "string" && event.timestamp !== undefined) event.timestamp = Math.round(event.timestamp);
            controller.enqueue(encoder.encode(`data: ${typeof event === "string" ? event : JSON.stringify(event)}\n\n`));
          };
          try {
            await produce(send, signal);
          } catch {
            // An unreadable run just ends the stream; the viewer says what it knows.
          }
          controller.close();
        },
      }),
      { headers: SSE_HEADERS },
    );
  }

  /** The run's steps and their traces, followed until the run ends. */
  function followTraces(runFolder: string, signal?: AbortSignal) {
    return stream(async (send) => {
      const started = await stat(runFolder);
      const runId = path.basename(runFolder);
      send({ type: "RUN_STARTED", threadId: runId, runId, timestamp: started.birthtimeMs || started.ctimeMs });
      const followed = new Map<string, { readLines?: () => Promise<string[]>; done: boolean; lastAt?: number }>();
      let previousDoneAt: number | undefined;

      async function catchUp(steps: Step[], running: boolean) {
        for (const [index, step] of steps.entries()) {
          let state = followed.get(step.key);
          if (!state) {
            state = { done: false };
            followed.set(step.key, state);
            send({ type: "STEP_STARTED", stepName: stepName(step), timestamp: step.firstAt ?? previousDoneAt ?? step.doneAt });
          }
          if (step.trace) {
            state.readLines ??= lineReader(roots, step.trace);
            for (const line of await state.readLines()) {
              let message: any;
              try {
                message = JSON.parse(line);
              } catch {
                continue;
              }
              for (const event of messageEvents(message)) {
                send(event);
                state.lastAt = event.timestamp ?? state.lastAt;
              }
            }
          }
          // Done once its output is written, a later step has started, or the run is over.
          if (!state.done && (step.doneAt !== undefined || index < steps.length - 1 || !running)) {
            state.done = true;
            send({ type: "STEP_FINISHED", stepName: stepName(step), timestamp: step.doneAt ?? state.lastAt });
          }
          previousDoneAt = step.doneAt ?? state.lastAt ?? previousDoneAt;
        }
      }

      while (!signal?.aborted) {
        const running = await isRunning(runFolder);
        await catchUp(await scanSteps(runFolder), running);
        if (!running) break;
        await sleep();
      }
      if (signal?.aborted) return;
      const ending = await runEnding(runFolder);
      // When the runner exited, so the run's time includes what came after its last trace (e.g. final reports).
      const timestamp = ending?.endedAt;
      if (ending?.error) send({ type: "RUN_ERROR", message: ending.error, timestamp, ...(ending.stopped ? { code: "stopped" } : {}) });
      else send({ type: "RUN_FINISHED", threadId: runId, runId, result: ending?.exitCode != null ? { exit_code: ending.exitCode } : {}, timestamp });
    }, signal);
  }

  /** A run recorded before traces existed: its AG-UI log, replayed and followed like the run's traces. */
  function replayLegacyLog(runFolder: string, signal?: AbortSignal) {
    return stream(async (send) => {
      const readLines = lineReader(roots, path.join(runFolder, LEGACY_LOG));
      while (!signal?.aborted) {
        const running = await isRunning(runFolder);
        for (const line of await readLines()) {
          send(line);
          if (/"type":"RUN_(FINISHED|ERROR)"/.test(line)) return;
        }
        if (!running) return;
        await sleep();
      }
    }, signal);
  }

  return async function events(url: URL, request: Request): Promise<Response> {
    const query = parseEventsQuery(url.searchParams);
    if (!query.ok) return new Response(query.error, { status: 400 });
    const runFolder = await roots.within(query.value.run);
    if (!runFolder || !(await stat(runFolder).then((s) => s.isDirectory(), () => false))) {
      return new Response("Only a run's folder inside the runs folders can be followed", { status: 400 });
    }
    const hasTraces = await stat(path.join(runFolder, TRACES_DIR)).then(() => true, () => false);
    const hasLegacyLog = await stat(path.join(runFolder, LEGACY_LOG)).then(() => true, () => false);
    return !hasTraces && hasLegacyLog ? replayLegacyLog(runFolder, request.signal) : followTraces(runFolder, request.signal);
  };
}
