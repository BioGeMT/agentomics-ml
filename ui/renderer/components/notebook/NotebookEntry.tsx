import type { Entry } from "@/model/run";
import { NotebookRecord } from "../records/NotebookRecord";
import { EntryHeader } from "./EntryHeader";
import { LiveLine } from "./LiveLine";

/** One step of an iteration: its header, then what the agent did, then what it is doing now. */
export function NotebookEntry({ entry, now, onPath }: { entry: Entry; now: number; onPath?: (path: string) => void }) {
  return (
    <article id={`entry-${entry.key}`} data-key={entry.key} className={`entry ${entry.status}`}>
      <EntryHeader entry={entry} now={now} />
      {entry.records
        .filter((record) => !(record.kind === "thinking" && record.endedAt !== undefined)) // finished reasoning is private
        .map((record, i, shown) => <NotebookRecord key={record.id} record={record} previous={shown[i - 1]} onPath={onPath} />)}
      {entry.status === "running" && <LiveLine entry={entry} now={now} />}
    </article>
  );
}
