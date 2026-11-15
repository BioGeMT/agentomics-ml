import type { EntryRecord } from "@/model/run";
import { formatClock } from "@/model/format";
import { recordTime } from "@/model/tool-calls";
import { RecordBody, type OpenPath } from "./RecordBody";

/** The margin only repeats a time when it changes from the record above. */
export function NotebookRecord({ record, previous, onPath }: { record: EntryRecord; previous?: EntryRecord; onPath?: OpenPath }) {
  // A run of tool calls reads as one list, so only its first row carries a time.
  const isStep = (r?: EntryRecord) => r?.kind === "tool" && r.name !== "final_result";
  const show = !previous || (formatClock(recordTime(previous)) !== formatClock(recordTime(record)) && !(isStep(record) && isStep(previous)));
  return <RecordBody record={record} show={show} onPath={onPath} />;
}
