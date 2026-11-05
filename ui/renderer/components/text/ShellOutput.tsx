import { segmentOutput } from "@/model/text-kinds";
import { CsvTable } from "./CsvTable";

/** A command's output: CSV blocks as tables, Python warnings folded, the rest as text. */
export function ShellOutput({ text }: { text: string }) {
  const segments = segmentOutput(text);
  if (segments.length === 0) return <div className="output empty">No output</div>;
  return (
    <div className="shell-output">
      {segments.map((segment, i) =>
        segment.kind === "csv" ? (
          <CsvTable key={i} text={segment.lines.join("\n")} limit={100} />
        ) : segment.kind === "warning" ? (
          <details key={i} className="shell-warning">
            <summary>{segment.lines.filter((l) => /Warning: /.test(l)).length || 1} warning{segment.lines.filter((l) => /Warning: /.test(l)).length > 1 ? "s" : ""}: {(/(\w*Warning): (.*)$/.exec(segment.lines[0])?.slice(1).join(": ") ?? segment.lines[0]).slice(0, 120)}</summary>
            <pre>{segment.lines.join("\n")}</pre>
          </details>
        ) : (
          <pre key={i} className="shell-text">{segment.lines.join("\n")}</pre>
        ),
      )}
    </div>
  );
}
