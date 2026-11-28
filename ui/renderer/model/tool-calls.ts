// Reading the agent's tool calls: their arguments, the files they touched and the workspace paths in their text.
import type { EntryRecord, ToolRecord } from "./run";

export function parseArgs(args: string): Record<string, any> {
  try {
    return JSON.parse(args);
  } catch {
    return {};
  }
}

export function fileName(path?: string): { name: string; dir: string } {
  if (!path) return { name: "", dir: "" };
  const parts = String(path).split("/");
  return { name: parts.pop() ?? "", dir: parts.join("/") };
}

export function recordTime(record: EntryRecord): number | undefined {
  return record.kind === "text" ? record.at : record.startedAt;
}

/** The workspace file a tool call wrote, ran or edited, if any. */
export function filePath(tool: ToolRecord): string | undefined {
  const args = parseArgs(tool.args);
  const path = tool.name === "run_python" ? args.python_file_path : ["write_python", "replace"].includes(tool.name) ? args.file_path : undefined;
  return typeof path === "string" && path.startsWith("/workspace/run/") ? path : undefined;
}

// Only the run folder is browsable; dataset paths (/workspace/prepared_datasets/…) stay plain text.
export const WORKSPACE_PATH = /\/workspace\/run\/[^\s"'`,;()<>\[\]]+/g;

/** What a tool call did, in one line: a verb and the thing it acted on. */
export function describeTool(tool: ToolRecord): { verb: string; detail: string; mono: boolean } {
  const args = parseArgs(tool.args);
  switch (tool.name) {
    case "bash":
      return { verb: "Shell", detail: String(args.command ?? "").split("\n")[0], mono: true };
    case "run_python":
      return { verb: "Ran", detail: fileName(args.python_file_path).name, mono: true };
    case "write_python":
      return { verb: "Wrote", detail: fileName(args.file_path).name, mono: true };
    case "replace":
      return { verb: "Edited", detail: fileName(args.file_path).name, mono: true };
    default:
      return { verb: tool.name.replaceAll("_", " "), detail: "", mono: false };
  }
}
