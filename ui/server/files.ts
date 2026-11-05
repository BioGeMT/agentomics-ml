// GET /api/files: a run's workspace, read-only (see ui/contract/protocol.ts).
import { parseFilesQuery, type DirEntry, type FilesResponse } from "@agentomics/contract/protocol";
import { createReadStream } from "node:fs";
import { spawn } from "node:child_process";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { followContainerLink, locate, type DatasetFor } from "./container-links";
import type { Roots } from "./roots";

const MAX_TEXT = 1024 * 1024; // larger files are summarised, not sent
const HIDDEN = new Set([".git"]);
const IMAGE_TYPES: Record<string, string> = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml" };
const RAW_TYPES: Record<string, string> = { ...IMAGE_TYPES, ".pdf": "application/pdf" };

const json = (body: FilesResponse, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** A folder's entries; a container link shows as what it points to in the run's dataset folder. */
async function listing(folder: string, runFolder: string, datasetFor?: DatasetFor): Promise<FilesResponse> {
  const names = (await readdir(folder, { withFileTypes: true })).filter((d) => !HIDDEN.has(d.name));
  const entries: DirEntry[] = await Promise.all(
    names.map(async (d) => {
      const entry = path.join(folder, d.name);
      const followed = d.isSymbolicLink() ? await followContainerLink(entry, runFolder, datasetFor) : null;
      const s = await stat(followed?.real ?? entry).catch(() => null);
      return { name: d.name, dir: Boolean(s?.isDirectory()), size: s?.size ?? 0, modified: s?.mtimeMs ?? 0 };
    }),
  );
  entries.sort((a, b) => Number(b.dir) - Number(a.dir) || a.name.localeCompare(b.name));
  return { kind: "dir", entries };
}

/** Images get a locked-down policy; PDFs need the built-in viewer, so no CSP that would block it. */
async function raw(file: string, type: string): Promise<Response> {
  const headers: Record<string, string> = { "Content-Type": type };
  if (IMAGE_TYPES[path.extname(file).toLowerCase()]) headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'";
  return new Response(new Uint8Array(await readFile(file)), { headers });
}

/** Content-Disposition that saves under name (ASCII fallback first, then the exact UTF-8 name). */
function saveAs(name: string): string {
  return `attachment; filename="${name.replace(/[^\x20-\x7e]|["\\]/g, "_")}"; filename*=UTF-8''${encodeURIComponent(name)}`;
}

/** The file itself, streamed from disk, for the browser or the desktop app to save. */
function attachment(file: string, size: number): Response {
  return new Response(Readable.toWeb(createReadStream(file)) as ReadableStream, {
    headers: { "Content-Type": "application/octet-stream", "Content-Length": String(size), "Content-Disposition": saveAs(path.basename(file)) },
  });
}

/**
 * A folder as a zip, streamed by the system's zip as it is written; nothing is kept on disk. Windows has
 * no zip command, but its bundled tar (bsdtar) writes zip archives.
 */
function zipped(folder: string): Response {
  const name = path.basename(folder);
  const [command, args] = process.platform === "win32" ? ["tar", ["--format", "zip", "-cf", "-", name]] : ["zip", ["-q", "-r", "-", name]];
  const child = spawn(command, args, { cwd: path.dirname(folder), stdio: ["ignore", "pipe", "ignore"], windowsHide: true });
  return new Response(Readable.toWeb(child.stdout) as ReadableStream, {
    headers: { "Content-Type": "application/zip", "Content-Disposition": saveAs(`${path.basename(folder)}.zip`) },
  });
}

async function content(file: string, size: number): Promise<FilesResponse> {
  const ext = path.extname(file).toLowerCase();
  if (IMAGE_TYPES[ext]) return { kind: "image", size };
  if (ext === ".pdf") return { kind: "pdf", size };
  if (size > MAX_TEXT) return { kind: "large", size };
  const data = await readFile(file);
  if (data.subarray(0, 8192).includes(0)) return { kind: "binary", size };
  return { kind: "text", size, text: data.toString("utf8") };
}

const OUTSIDE = "Only files inside a run's folder can be opened";
const ERRORS = {
  outside: { status: 403, error: OUTSIDE },
  missing: { status: 404, error: "This file doesn't exist (yet)." },
  "container-link": { status: 404, error: "This is a link the agent made inside its container; it doesn't point to anything on this Mac." },
} as const;

export function createFiles(roots: Roots, datasetFor?: DatasetFor) {
  return async function files(url: URL): Promise<Response> {
    const query = parseFilesQuery(url.searchParams);
    if (!query.ok) return json({ error: query.error }, 400);
    const { root, path: relative, raw: wantsRaw, download } = query.value;
    const realRoot = await roots.within(root);
    const located = realRoot ? await locate(realRoot, relative, datasetFor) : ({ ok: false, reason: "outside" } as const);
    if (!located.ok) return json({ error: ERRORS[located.reason].error }, ERRORS[located.reason].status);
    const target = located.real;
    const info = await stat(target);
    if (download) return info.isDirectory() ? zipped(target) : attachment(target, info.size);
    if (info.isDirectory()) return json(await listing(target, realRoot!, datasetFor));
    const rawType = RAW_TYPES[path.extname(target).toLowerCase()];
    if (wantsRaw && rawType) return raw(target, rawType);
    return json(await content(target, info.size));
  };
}
