import { useStepOutput } from "@/hooks/useStepOutput";
import type { Entry, EntryRecord, ToolRecord } from "@/model/run";
import { formatClock } from "@/model/format";
import { recordTime } from "@/model/tool-calls";
import { Margin } from "../records/Margin";
import { NotebookRecord } from "../records/NotebookRecord";
import { sameCall, ToolRun } from "../records/ToolRun";
import { EntryHeader } from "./EntryHeader";
import { LiveLine } from "./LiveLine";

/** One step of an iteration: its header, then what the agent did, then what it is doing now. */
export function NotebookEntry({ entry, workspace, now, onPath }: { entry: Entry; workspace: string | null; now: number; onPath?: (path: string) => void }) {
  const saved = useStepOutput(workspace, entry);
  return (
    <article id={`entry-${entry.key}`} data-key={entry.key} className={`entry ${entry.status}`}>
      <EntryHeader entry={entry} now={now} />
      {runs(entry.records.filter(shown))
        .map((run, i, shown) => {
          const previous = shown[i - 1]?.at(-1);
          if (run.length === 1) return <NotebookRecord key={run[0].id} record={run[0]} previous={previous} saved={saved} onPath={onPath} />;
          const show = !previous || (previous.kind !== "tool" && formatClock(recordTime(previous)) !== formatClock(recordTime(run[0])));
          return (
            <div key={run[0].id} className="record record-step">
              <Margin at={recordTime(run[0])} show={show} />
              <ToolRun tools={run as ToolRecord[]} onPath={onPath} />
            </div>
          );
        })}
      {entry.status === "running" && <LiveLine entry={entry} now={now} />}
    </article>
  );
}

/** Records with something to show: finished reasoning is private, and empty messages say nothing. */
function shown(record: EntryRecord): boolean {
  if (record.kind === "thinking") return record.endedAt === undefined;
  return record.kind !== "text" || record.text.trim() !== "";
}

/** The records in order, consecutive identical tool calls together. */
function runs(records: EntryRecord[]): EntryRecord[][] {
  const result: EntryRecord[][] = [];
  for (const record of records) {
    const last = result.at(-1)?.at(-1);
    if (last?.kind === "tool" && record.kind === "tool" && sameCall(last, record)) result.at(-1)!.push(record);
    else result.push([record]);
  }
  return result;
}
