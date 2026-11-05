// @agentomics/server: answers the agentomics:// protocol's endpoints (ui/contract/protocol.ts) with
// web Responses. The renderer's Vite dev server (ui/renderer/dev-api.ts) and the desktop app's
// protocol handler (ui/desktop/src/protocol.ts) both serve runs through it, so they behave the same.
import type { DatasetFor } from "./container-links";
import { createEvents, type IsRunning } from "./events";
import { createFiles } from "./files";
import { createRoots } from "./roots";

export type WorkspaceServer = {
  /** POST /api/events */
  events(url: URL, request: Request): Promise<Response>;
  /** GET /api/files */
  files(url: URL): Promise<Response>;
};

export type { DatasetFor, IsRunning };

/**
 * roots: the folders runs live in (the repo's outputs/, or the app's runs folder). datasetFor: the dataset
 * folder of a run, from the app's run history, so links the agent made to its datasets open. isRunning: whether
 * a run is still going, so its events are followed until it ends; without it, logs are read once.
 */
export function createWorkspaceServer({ roots, datasetFor, isRunning }: { roots: string[]; datasetFor?: DatasetFor; isRunning?: IsRunning }): WorkspaceServer {
  const allowed = createRoots(roots);
  return { events: createEvents(allowed, isRunning), files: createFiles(allowed, datasetFor) };
}
