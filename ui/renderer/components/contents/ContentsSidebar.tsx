import type { Entry, Run } from "@/model/run";
import { ChapterHeader } from "./ChapterHeader";
import { ContentsEntry } from "./ContentsEntry";
import { ContentsSkeleton } from "./ContentsSkeleton";

/** The left sidebar: every iteration's steps, each with its state and time; the one being read is highlighted. */
export function ContentsSidebar({ run, chapters, folded, active, current, now, hidden, inert, onToggleIteration, onGoTo }: {
  run: Run;
  chapters: [number, Entry[]][];
  folded: Set<number>;
  active: string | null;
  /** The step a live run is on, or about to start. */
  current?: Entry;
  now: number;
  hidden: boolean;
  inert: boolean;
  onToggleIteration: (iteration: number, event: React.MouseEvent) => void;
  onGoTo: (key: string) => void;
}) {
  return (
    <nav className="contents" id="panel-contents" aria-label="Contents" hidden={hidden} inert={inert}>
      <div className="side-head">
        <h2>Contents</h2>
      </div>
      {chapters.length === 0 && run.status === "connecting" && <ContentsSkeleton />}
      {chapters.map(([iteration, chapter]) => (
        <section key={iteration} className="chapter">
          <ChapterHeader
            iteration={iteration}
            snapshot={run.snapshots.find((s) => s.iteration + 1 === iteration)}
            open={!folded.has(iteration)}
            live={current?.iteration === iteration}
            onToggle={(event) => onToggleIteration(iteration, event)}
          />
          <ol hidden={folded.has(iteration)}>
            {chapter.map((entry) => (
              <li key={entry.key}>
                <ContentsEntry entry={entry} reading={active === entry.key} starting={current?.key === entry.key && entry.status === "upcoming"} now={now} onGoTo={() => onGoTo(entry.key)} />
              </li>
            ))}
          </ol>
        </section>
      ))}
    </nav>
  );
}
