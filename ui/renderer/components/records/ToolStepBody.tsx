import { LoaderCircle } from "lucide-react";
import type { ToolRecord } from "@/model/run";
import { printedFile } from "@/model/text-kinds";
import { parseArgs } from "@/model/tool-calls";
import { CodeBlock } from "../text/CodeBlock";
import { ShellCommand } from "../text/ShellCommand";
import { Output } from "./Output";
import { ProcessOutput } from "./ProcessOutput";

/** An opened tool call: the command, code or edit it made, and what it printed. */
export function ToolStepBody({ tool }: { tool: ToolRecord }) {
  const args = parseArgs(tool.args);
  const pending = tool.result === undefined;
  const output = pending ? <div className="output empty"><LoaderCircle size={14} className="spin" aria-hidden /> Running…</div> : <Output text={tool.result} />;
  switch (tool.name) {
    case "bash":
      return (
        <div className="step-body">
          <div className="command"><ShellCommand command={String(args.command ?? "")} /></div>
          {pending ? output : printedFile(String(args.command ?? "")) ? (
            <Output text={tool.result} name={printedFile(String(args.command ?? ""))} />
          ) : (
            <ProcessOutput text={tool.result} />
          )}
        </div>
      );
    case "run_python":
      return <div className="step-body">{pending ? output : <ProcessOutput text={tool.result} />}</div>;
    case "write_python":
      return (
        <div className="step-body">
          <CodeBlock text={String(args.code ?? "")} language="python" />
        </div>
      );
    case "replace":
      return (
        <div className="step-body">
          <pre className="diff">
            {String(args.old ?? "").split("\n").map((line, i) => <span key={`o${i}`} className="del">- {line}{"\n"}</span>)}
            {String(args.new ?? "").split("\n").map((line, i) => <span key={`n${i}`} className="add">+ {line}{"\n"}</span>)}
          </pre>
        </div>
      );
    default:
      return (
        <div className="step-body">
          <pre className="command">{JSON.stringify(args, null, 2)}</pre>
          {output}
        </div>
      );
  }
}
