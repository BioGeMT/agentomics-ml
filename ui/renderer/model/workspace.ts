// Reading a run's workspace through GET /api/files (ui/contract/protocol.ts).
import { filesUrl, type FilesQuery, type FilesResponse } from "@agentomics/contract/protocol";

/** A listing, a file's content or summary, or an error in words (fallback when the server gives none). */
export async function readWorkspace(query: FilesQuery, fallback = "Couldn't read this."): Promise<FilesResponse> {
  try {
    const response = await fetch(filesUrl(query));
    const body: FilesResponse = await response.json();
    return response.ok ? body : { error: ("error" in body && body.error) || fallback };
  } catch {
    return { error: fallback };
  }
}

/** A text file's content, or null when it is missing, too large or not text. */
export async function readText(root: string, path: string): Promise<string | null> {
  const body = await readWorkspace({ root, path });
  return "kind" in body && body.kind === "text" ? body.text : null;
}

/** Whether a path exists in the workspace. */
export async function exists(root: string, path: string): Promise<boolean> {
  return !("error" in (await readWorkspace({ root, path })));
}

/** The URL of an image or PDF's bytes, for <img> and <iframe>. */
export function rawUrl(root: string, path: string): string {
  return filesUrl({ root, path, raw: true });
}

/** The URL that streams a file as an attachment, for saving it. */
export function downloadUrl(root: string, path: string): string {
  return filesUrl({ root, path, download: true });
}
