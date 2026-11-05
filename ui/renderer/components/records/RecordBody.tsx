import type { EntryRecord } from "@/model/run";
import { parseArgs } from "@/model/tool-calls";
import { LinkedText } from "./LinkedText";
import { Margin } from "./Margin";
import { SignedResult } from "./SignedResult";
import { ToolStep } from "./ToolStep";

export type OpenPath = (path: string) => void;

/** One record in an entry: a note, the step's signed result, or a tool call. */
export function RecordBody({ record, show, onPath }: { record: EntryRecord; show: boolean; onPath?: OpenPath }) {
  if (record.kind === "text") {
    if (!record.text.trim()) return null;
    return (
      <div className="record">
        <Margin at={record.at} show={show} />
        <p className="note"><LinkedText text={record.text} onPath={onPath} /></p>
      </div>
    );
  }
  // Reasoning is private and says nothing on its own; the entry's live line shows when it is happening.
  if (record.kind === "thinking") return null;
  if (record.name === "final_result") {
    return (
      <div className="record">
        <Margin at={record.startedAt} label="Result" />
        <SignedResult args={parseArgs(record.args)} at={record.startedAt} onPath={onPath} />
      </div>
    );
  }
  return (
    <div className="record record-step">
      <Margin at={record.startedAt} show={show} />
      <ToolStep tool={record} onPath={onPath} />
    </div>
  );
}
