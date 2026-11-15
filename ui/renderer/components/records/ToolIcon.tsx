import { FileCode, FilePen, PenLine, Play, SquareTerminal } from "lucide-react";

/** The icon for a tool: a terminal for shell commands, play for running a script, and so on. */
export function ToolIcon({ name }: { name: string }) {
  switch (name) {
    case "bash":
      return <SquareTerminal size={14} aria-hidden />;
    case "run_python":
      return <Play size={14} aria-hidden />;
    case "write_python":
      return <FileCode size={14} aria-hidden />;
    case "replace":
      return <FilePen size={14} aria-hidden />;
    default:
      return <PenLine size={14} aria-hidden />;
  }
}
