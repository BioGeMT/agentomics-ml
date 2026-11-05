// The dev server's /api/events and /api/files (ui/contract/protocol.ts), answered by @agentomics/server
// over the repo's outputs/ - the same code the desktop app's agentomics:// protocol uses once built.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin, ViteDevServer } from "vite";
import { ENDPOINTS } from "../contract/protocol.ts";

type WorkspaceServer = { events(url: URL, request: Request): Promise<Response>; files(url: URL): Promise<Response> };

const OUTPUTS = path.resolve(import.meta.dirname, "..", "..", "outputs");
// The development app's run history (ui/desktop/src/history.ts), which records each run's dataset folder and process.
const HISTORY = path.join(os.homedir(), "Library", "Application Support", "agentomics-desktop", "runs.json");

type HistoryEntry = { workspace: string; folder: string; status: string; pid?: number };

function historyEntry(runFolder: string): HistoryEntry | undefined {
  try {
    const runs: HistoryEntry[] = JSON.parse(fs.readFileSync(HISTORY, "utf8"));
    return runs.find((r) => fs.existsSync(r.workspace) && fs.realpathSync(r.workspace) === runFolder);
  } catch {
    return undefined;
  }
}

/** A run's dataset folder from the development app's history, so links the agent made into it open. */
async function datasetFor(runFolder: string): Promise<string | null> {
  return historyEntry(runFolder)?.folder ?? null;
}

/** Whether the development app's run in runFolder is still going, so its event log is followed. */
async function isRunning(runFolder: string): Promise<boolean> {
  const entry = historyEntry(runFolder);
  if (entry?.status !== "running" || !entry.pid) return false;
  try {
    process.kill(entry.pid, 0);
    return true;
  } catch (error: any) {
    return error?.code === "EPERM";
  }
}
// The dev server listens on 127.0.0.1 only, but a web page could still reach it through DNS rebinding,
// so requests must name this server as their host.
const THIS_SERVER = /^(127\.0\.0\.1|localhost):3000$/;

export function devApi(): Plugin {
  let workspace: WorkspaceServer | null = null;
  // Loaded through Vite so the server's own @agentomics/* imports resolve.
  const load = async (server: ViteDevServer) => {
    const { createWorkspaceServer } = await server.ssrLoadModule(path.resolve(import.meta.dirname, "../server/index.ts"));
    return (workspace ??= createWorkspaceServer({ roots: [OUTPUTS], datasetFor, isRunning }) as WorkspaceServer);
  };

  return {
    name: "agentomics-dev-api",
    configureServer(server) {
      server.middlewares.use(async (req: IncomingMessage, res: ServerResponse, next: (error?: unknown) => void) => {
        const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "127.0.0.1:3000"}`);
        if (url.pathname !== ENDPOINTS.events && url.pathname !== ENDPOINTS.files) return next();
        if (!THIS_SERVER.test(req.headers.host ?? "")) {
          res.writeHead(403).end("Forbidden");
          return;
        }
        try {
          const ws = await load(server);
          // A closed tab stops following a run's event log.
          const controller = new AbortController();
          res.on("close", () => { if (!res.writableEnded) controller.abort(); });
          const request = new Request(url, {
            method: req.method,
            headers: req.headers as HeadersInit,
            body: req.method === "POST" ? (Readable.toWeb(req) as ReadableStream) : undefined,
            signal: controller.signal,
            // Node's fetch Request needs this for a streamed body.
            ...({ duplex: "half" } as object),
          });
          const response = url.pathname === ENDPOINTS.events ? await ws.events(url, request) : await ws.files(url);
          res.writeHead(response.status, Object.fromEntries(response.headers));
          res.flushHeaders();
          if (response.body) Readable.fromWeb(response.body as import("node:stream/web").ReadableStream).pipe(res);
          else res.end();
        } catch (error) {
          next(error);
        }
      });
    },
  };
}
