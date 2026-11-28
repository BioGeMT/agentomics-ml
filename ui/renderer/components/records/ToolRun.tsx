import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { formatDuration } from "@/model/format";
import type { ToolRecord } from "@/model/run";
import { describeTool } from "@/model/tool-calls";
import type { OpenPath } from "./RecordBody";
import { ToolIcon } from "./ToolIcon";
import { ToolStep } from "./ToolStep";

/**
 * The same call made several times in a row (e.g. edit after edit to one script): one row with the count
 * and their time together, opening to each call.
 */
export function ToolRun({ tools, onPath }: { tools: ToolRecord[]; onPath?: OpenPath }) {
  const [open, setOpen] = useState(false);
  const { verb, detail, mono } = describeTool(tools[0]);
  const time = tools.reduce((sum, t) => sum + (t.endedAt !== undefined && t.startedAt !== undefined ? t.endedAt - t.startedAt : 0), 0);
  return (
    <div className={`step tool-run ${open ? "open" : ""}`}>
      <button className="step-row" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="step-icon"><ToolIcon name={tools[0].name} /></span>
        <span className="step-verb">{verb}</span>
        {detail && <span className={`step-detail ${mono ? "mono" : ""}`}>{detail}</span>}
        <span className="run-count" aria-label={`${tools.length} times`}>×{tools.length}</span>
        <span className="step-time">{formatDuration(time)}</span>
        <ChevronRight size={14} className={`disclosure ${open ? "open" : ""}`} aria-hidden />
      </button>
      {open && (
        <div className="tool-run-calls">
          {tools.map((tool) => <ToolStep key={tool.id} tool={tool} onPath={onPath} />)}
        </div>
      )}
    </div>
  );
}

/** Whether two tool calls read the same in the notebook, so they can share a row. */
export function sameCall(a: ToolRecord, b: ToolRecord): boolean {
  if (a.name !== b.name || a.name === "final_result" || a.result === undefined || b.result === undefined) return false;
  const x = describeTool(a);
  const y = describeTool(b);
  return x.verb === y.verb && x.detail === y.detail;
}
