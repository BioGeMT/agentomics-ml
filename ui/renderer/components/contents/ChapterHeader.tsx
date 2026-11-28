import { GitFork, LoaderCircle } from "lucide-react";
import type { Snapshot } from "@/model/run";
import { Disclosure } from "../notebook/Disclosure";
import { ChapterMeta } from "./ChapterMeta";

/** Contents: the iteration header folds its list of steps; the live iteration carries a spinner. Pointed at, a
 * finished iteration offers Fork in place of its score. */
export function ChapterHeader({ iteration, snapshot, open, live, onToggle, forkHref }: {
  iteration: number;
  snapshot?: Snapshot;
  open: boolean;
  live?: boolean;
  onToggle: (event: React.MouseEvent) => void;
  forkHref?: string;
}) {
  return (
    <h3>
      <button className="chapter-toggle" aria-expanded={open} onClick={onToggle} title="Option-click to fold or unfold every iteration">
        <Disclosure open={open} />
        Iteration {iteration}
        {live && <LoaderCircle size={12} className="mark running spin chapter-live" aria-label="in progress" />}
        <ChapterMeta snapshot={snapshot} />
      </button>
      {forkHref && (
        <a className="contents-fork" href={forkHref} title={`Fork from iteration ${iteration}: keep iterations 1–${iteration} and try something different`}>
          <GitFork size={11} strokeWidth={2.2} aria-hidden /> Fork
        </a>
      )}
    </h3>
  );
}
