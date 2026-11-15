import { ArrowUpRight, ChevronRight, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { formatDuration } from "@/model/format";
import type { ToolRecord } from "@/model/run";
import { describeTool, filePath } from "@/model/tool-calls";
import type { OpenPath } from "./RecordBody";
import { ToolIcon } from "./ToolIcon";
import { ToolStepBody } from "./ToolStepBody";

/** A tool call: a quiet one-line row that opens to the command, the code and the output. */
export function ToolStep({ tool, onPath }: { tool: ToolRecord; onPath?: OpenPath }) {
  const [open, setOpen] = useState(false);
  const pending = tool.result === undefined;
  const { verb, detail, mono } = describeTool(tool);
  return (
    <div className={`step ${open ? "open" : ""} ${pending ? "pending" : ""}`}>
      <button className="step-row" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="step-icon">{pending ? <LoaderCircle size={14} className="spin" aria-hidden /> : <ToolIcon name={tool.name} />}</span>
        <span className="step-verb">{verb}</span>
        {detail && <span className={`step-detail ${mono ? "mono" : ""}`}>{detail}</span>}
        <span className="step-time">{pending ? "running" : tool.endedAt !== undefined && tool.startedAt !== undefined ? formatDuration(tool.endedAt - tool.startedAt) : ""}</span>
        <ChevronRight size={14} className={`disclosure ${open ? "open" : ""}`} aria-hidden />
      </button>
      {onPath && filePath(tool) && (
        <button className="step-open" onClick={() => onPath(filePath(tool)!)} aria-label={`Open ${detail}`} title={`Open ${detail}`}>
          <ArrowUpRight size={13} aria-hidden />
        </button>
      )}
      {open && <ToolStepBody tool={tool} />}
    </div>
  );
}
