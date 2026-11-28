import { WORKSPACE_PATH } from "@/model/tool-calls";
import type { OpenPath } from "./RecordBody";

// The inline markdown agents write in their notes and results: `code` and **bold**.
const INLINE = /`([^`\n]+)`|\*\*([^*\n]+)\*\*/g;

/**
 * The agent's prose: its `code` and **bold** shown as such, and every /workspace/run/… path a link that
 * opens the file.
 */
export function LinkedText({ text, onPath }: { text: string; onPath?: OpenPath }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const at = match.index ?? 0;
    parts.push(...withPaths(text.slice(last, at), onPath, `${last}`));
    const [, code, bold] = match;
    parts.push(code !== undefined
      ? <code key={`c${at}`} className="inline-code">{withPaths(code, onPath, `c${at}`)}</code>
      : <strong key={`b${at}`}>{withPaths(bold, onPath, `b${at}`)}</strong>);
    last = at + match[0].length;
  }
  parts.push(...withPaths(text.slice(last), onPath, `${last}`));
  return <>{parts}</>;
}

/** Text with its /workspace/run/… paths as links (when the run's files can be opened). */
function withPaths(text: string, onPath: OpenPath | undefined, key: string): React.ReactNode[] {
  if (!onPath || !text.includes("/workspace/run/")) return [text];
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(WORKSPACE_PATH)) {
    const path = match[0].replace(/[.:]+$/, "");
    const at = match.index ?? 0;
    parts.push(text.slice(last, at));
    parts.push(<button key={`${key}-${at}`} className="path-link" onClick={() => onPath(path)} title="Open in the file viewer">{path}</button>);
    last = at + path.length;
  }
  parts.push(text.slice(last));
  return parts;
}
