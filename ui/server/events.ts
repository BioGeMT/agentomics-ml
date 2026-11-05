// POST /api/events: a run's AG-UI events as text/event-stream (see ui/contract/protocol.ts).
// The run appends them to <run>/logs/ui_events.jsonl; this reads that file as it grows.
import { parseEventsQuery } from "@agentomics/contract/protocol";
import { open, realpath, stat } from "node:fs/promises";
import path from "node:path";
import type { Roots } from "./roots";

const SSE_HEADERS = { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" };
const LOG_IN_RUN = path.join("logs", "ui_events.jsonl");
const POLL_MS = 500;

/** Whether a run is still going, from the app's run history; its log may still grow while it is. */
export type IsRunning = (runFolder: string) => Promise<boolean>;

const ends = (line: string) => {
  try {
    const { type } = JSON.parse(line);
    return type === "RUN_FINISHED" || type === "RUN_ERROR";
  } catch {
    return false;
  }
};

export function createEvents(roots: Roots, isRunning: IsRunning = async () => false) {
  /** The real path of a run's event log, which need not exist yet, if the request names one in a runs folder. */
  async function eventLog(file: string): Promise<string | null> {
    const resolved = path.resolve(file);
    const runFolder = path.dirname(path.dirname(resolved));
    if (resolved !== path.join(runFolder, LOG_IN_RUN)) return null;
    const real = await roots.within(runFolder);
    return real ? path.join(real, LOG_IN_RUN) : null;
  }

  /** Streams the log from the start, then follows it until it records the run's end or the run stops. */
  async function follow(file: string, request: Request): Promise<Response> {
    const found = await eventLog(file);
    if (!found) return new Response("Only a run's event log (<run>/logs/ui_events.jsonl) can be read", { status: 400 });
    const log = found;
    const runFolder = path.dirname(path.dirname(log));
    const signal = request.signal;
    const encoder = new TextEncoder();
    let position = 0;
    let decoder = new TextDecoder();
    let pending = "";

    /** Sends the lines appended since the last read; true once one records the run's end. */
    async function readMore(controller: ReadableStreamDefaultController): Promise<boolean> {
      if (!(await roots.within(log))) return false; // not there yet, or a link out of the runs folders
      const size = await stat(log).then((s) => s.size, () => null);
      if (size === null || size === position) return false;
      if (size < position) [position, pending, decoder] = [0, "", new TextDecoder()]; // the run started over
      const handle = await open(log);
      try {
        const { buffer, bytesRead } = await handle.read(Buffer.alloc(size - position), 0, size - position, position);
        position += bytesRead;
        pending += decoder.decode(buffer.subarray(0, bytesRead), { stream: true });
      } finally {
        await handle.close();
      }
      const lines = pending.split("\n");
      pending = lines.pop() ?? ""; // a line still being written
      for (const line of lines.filter((l) => l.trim())) {
        controller.enqueue(encoder.encode(`data: ${line}\n\n`));
        if (ends(line)) return true;
      }
      return false;
    }

    const stream = new ReadableStream({
      async start(controller) {
        try {
          while (!signal?.aborted) {
            const running = await isRunning(runFolder);
            if ((await readMore(controller)) || !running) break;
            await new Promise((resolve) => setTimeout(resolve, POLL_MS));
          }
        } catch {
          // An unreadable log just ends the stream; the viewer says the run stopped.
        }
        controller.close();
      },
    });
    return new Response(stream, { headers: SSE_HEADERS });
  }

  return async function events(url: URL, request: Request): Promise<Response> {
    const query = parseEventsQuery(url.searchParams);
    if (!query.ok) return new Response(query.error, { status: 400 });
    return follow(query.value.log, request);
  };
}
