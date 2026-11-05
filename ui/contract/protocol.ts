// The agentomics:// protocol: how the UI reads runs. One definition, used by the page that asks
// (ui/renderer), the server that answers (ui/server) and the desktop app that routes (ui/desktop).
//
//   agentomics://app/<page>/                the UI itself: static files built from ui/renderer
//
//   POST /api/events                        a run's AG-UI events, as text/event-stream
//        ?log=<run>/logs/ui_events.jsonl      the run's event log from the start; while the run is going the
//                                             stream follows the log until it records the run's end
//        → 200 stream · 400 bad query
//
//   GET  /api/files                         a run's workspace, read-only
//        ?root=<run folder>&path=<relative>   → FilesResponse (JSON): a folder listing, a text file, or a summary
//        ...&raw=1                            → the bytes of an image or PDF, for <img> and the PDF viewer
//        ...&download=1                       → a file streamed as an attachment to save; a folder as a zip
//        → 400 bad query · 403 outside the runs folders · 404 missing, or a container link with nowhere to go
//        Links the agent made inside its container (to /datasets/…) open the run's dataset folder on the Mac.
//
// Only files under the runs folders are served, checked after resolving symlinks. While developing,
// the renderer's Vite dev server answers the same endpoints at http://127.0.0.1:3000.
//
// A runtime part of @agentomics/contract: small pure functions, no imports.

export const ENDPOINTS = { events: "/api/events", files: "/api/files" } as const;

/** Where a run's events come from: its event log, which a run started with AGENTOMICS_UI_EVENTS=1 writes. */
export type EventsSource = { log: string };

export type FilesQuery = { root: string; path: string; raw?: boolean; download?: boolean };

export type DirEntry = { name: string; dir: boolean; size: number; modified: number };

export type FilesResponse =
  | { kind: "dir"; entries: DirEntry[] }
  | { kind: "text"; size: number; text: string }
  | { kind: "image" | "pdf" | "binary" | "large"; size: number }
  | { error: string };

type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

export function eventsUrl({ log }: EventsSource): string {
  return `${ENDPOINTS.events}?${new URLSearchParams({ log })}`;
}

export function parseEventsQuery(query: URLSearchParams): Parsed<EventsSource> {
  const log = query.get("log");
  return log ? { ok: true, value: { log } } : { ok: false, error: "Give a run's event log (?log=)" };
}

export function filesUrl({ root, path, raw, download }: FilesQuery): string {
  const query = new URLSearchParams({ root, path });
  if (raw) query.set("raw", "1");
  if (download) query.set("download", "1");
  return `${ENDPOINTS.files}?${query}`;
}

export function parseFilesQuery(query: URLSearchParams): Parsed<FilesQuery> {
  const root = query.get("root");
  if (!root) return { ok: false, error: "Missing root" };
  return { ok: true, value: { root, path: query.get("path") ?? "", raw: query.get("raw") === "1", download: query.get("download") === "1" } };
}
