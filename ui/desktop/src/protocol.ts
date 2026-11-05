// The agentomics:// protocol (defined in ui/contract/protocol.ts): how the desktop app serves its UI
// and run data without a web server. Electron hands every request from the window to the handler
// below; nothing listens on a port, and only this app can reach it.
//
//   agentomics://app/api/events   → @agentomics/server (a run's event stream)
//   agentomics://app/api/files    → @agentomics/server (a run's workspace)
//   agentomics://app/<anything>   → the exported UI in webRoot (ui/renderer/out)
import { ENDPOINTS } from "@agentomics/contract/protocol";
import { createWorkspaceServer, type DatasetFor, type IsRunning } from "@agentomics/server";
import { net, protocol } from "electron";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SCHEME = "agentomics";
const HOST = "app";
export const ORIGIN = `${SCHEME}://${HOST}`;

type Route = (url: URL, request: Request) => Promise<Response>;

/** Must run before the app is ready: lets the scheme behave like https (fetch, streaming, storage). */
export function registerScheme(): void {
  protocol.registerSchemesAsPrivileged([
    { scheme: SCHEME, privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, corsEnabled: true } },
  ]);
}

/**
 * webRoot: the built UI; runRoots: the folders runs live in, the only files /api/* will serve;
 * datasetFor: a run's dataset folder, which its container links point into; isRunning: whether a run is going.
 */
export function serveApp({ webRoot, runRoots, datasetFor, isRunning }: { webRoot: string; runRoots: string[]; datasetFor: DatasetFor; isRunning: IsRunning }): void {
  const workspace = createWorkspaceServer({ roots: runRoots, datasetFor, isRunning });
  const routes: Record<string, Route> = {
    [ENDPOINTS.events]: (url, request) => workspace.events(url, request),
    [ENDPOINTS.files]: (url) => workspace.files(url),
  };
  const ui = staticFiles(path.resolve(webRoot));

  protocol.handle(SCHEME, (request) => {
    const url = new URL(request.url);
    if (url.host !== HOST) return new Response("Not found", { status: 404 });
    const route = routes[url.pathname] ?? ui;
    return route(url, request);
  });
}

/** Serves the static export: the file, else its folder's index.html (trailingSlash export), else 404.html. */
function staticFiles(root: string): Route {
  function find(pathname: string): string | null {
    let relative: string;
    try {
      relative = decodeURIComponent(pathname).replace(/^\/+/, "");
    } catch {
      return null;
    }
    const base = path.resolve(root, relative);
    if (base !== root && !base.startsWith(root + path.sep)) return null;
    for (const candidate of [base, path.join(base, "index.html"), `${base}.html`]) {
      if (fs.statSync(candidate, { throwIfNoEntry: false })?.isFile()) return candidate;
    }
    return null;
  }

  return async (url) => {
    const file = find(url.pathname);
    if (file) return net.fetch(pathToFileURL(file).href);
    const missing = find("/404.html");
    return missing
      ? new Response(new Uint8Array(fs.readFileSync(missing)), { status: 404, headers: { "Content-Type": "text/html" } })
      : new Response("Not found", { status: 404 });
  };
}
