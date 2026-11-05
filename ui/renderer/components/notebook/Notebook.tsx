import type { EventsSource } from "@agentomics/contract/protocol";
import { Fragment, type RefObject } from "react";
import type { Entry, Run } from "@/model/run";
import { ChapterBreak } from "./ChapterBreak";
import { JumpToLive } from "./JumpToLive";
import { NotebookEntry } from "./NotebookEntry";
import { NextStepLine } from "./NextStepLine";
import { NotebookStatus } from "./NotebookStatus";
import { RunClosing } from "./RunClosing";

/** The page: the run's entries in order, each iteration opening with a chapter break that can fold it. */
export function Notebook({ pageRef, run, source, unreachable, folded, now, elapsed, done, inert, showJump, liveEntry, starting, onScroll, onJump, onConnect, onToggleIteration, onPath }: {
  pageRef: RefObject<HTMLElement | null>;
  run: Run;
  source: EventsSource | null;
  unreachable: string | null;
  folded: Set<number>;
  now: number;
  elapsed?: number;
  done: number;
  inert: boolean;
  showJump: boolean;
  liveEntry?: Entry;
  /** The step a live run is about to start, while it is between steps. */
  starting?: Entry;
  onScroll: () => void;
  onJump: () => void;
  onConnect: (source: EventsSource) => void;
  onToggleIteration: (iteration: number, event: React.MouseEvent) => void;
  onPath?: (path: string, entry: Entry) => void;
}) {
  return (
    <main className="page" id="panel-notebook" ref={pageRef} onScroll={onScroll} inert={inert}>
      <div className="sheet">
        <NotebookStatus run={run} source={source} unreachable={unreachable} now={now} onRetry={onConnect} />
        {run.entries.filter((e) => e.position > 0 || e.records.length).map((entry, index, shown) => (
          <Fragment key={entry.key + entry.startedAt}>
            {entry.iteration !== shown[index - 1]?.iteration && (
              <ChapterBreak
                iteration={entry.iteration}
                snapshot={run.snapshots.find((s) => s.iteration + 1 === entry.iteration)}
                entries={run.entries.filter((e) => e.iteration === entry.iteration)}
                open={!folded.has(entry.iteration)}
                onToggle={(event) => onToggleIteration(entry.iteration, event)}
              />
            )}
            {!folded.has(entry.iteration) && <NotebookEntry entry={entry} now={now} onPath={onPath ? (path) => onPath(path, entry) : undefined} />}
          </Fragment>
        ))}
        {starting && run.entries.some((e) => e.position > 0) && (
          <NextStepLine entry={starting} since={Math.max(...run.entries.map((e) => e.endedAt ?? e.startedAt ?? 0)) || run.startedAt} now={now} />
        )}
        {run.status === "finished" && <RunClosing done={done} elapsed={elapsed} />}
      </div>
      {showJump && liveEntry && <JumpToLive entry={liveEntry} onJump={onJump} />}
    </main>
  );
}
