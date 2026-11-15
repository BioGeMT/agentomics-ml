import { WORKSPACE_PATH } from "@/model/tool-calls";
import type { OpenPath } from "./RecordBody";

/** Text with every /workspace/… path turned into a link that opens the file. */
export function LinkedText({ text, onPath }: { text: string; onPath?: OpenPath }) {
  if (!onPath || !text.includes("/workspace/run/")) return <>{text}</>;
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(WORKSPACE_PATH)) {
    const path = match[0].replace(/[.:]+$/, "");
    const at = match.index ?? 0;
    parts.push(text.slice(last, at));
    parts.push(<button key={at} className="path-link" onClick={() => onPath(path)} title="Open in the file viewer">{path}</button>);
    last = at + path.length;
  }
  parts.push(text.slice(last));
  return <>{parts}</>;
}
