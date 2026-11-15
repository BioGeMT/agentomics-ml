import { LoaderCircle } from "lucide-react";
import type { Snapshot } from "@/model/run";
import { Disclosure } from "../notebook/Disclosure";
import { ChapterMeta } from "./ChapterMeta";

/** Contents: the iteration header folds its list of steps; the live iteration carries a spinner. */
export function ChapterHeader({ iteration, snapshot, open, live, onToggle }: {
  iteration: number;
  snapshot?: Snapshot;
  open: boolean;
  live?: boolean;
  onToggle: (event: React.MouseEvent) => void;
}) {
  return (
    <h3>
      <button className="chapter-toggle" aria-expanded={open} onClick={onToggle} title="Option-click to fold or unfold every iteration">
        <Disclosure open={open} />
        Iteration {iteration}
        {live && <LoaderCircle size={12} className="mark running spin chapter-live" aria-label="in progress" />}
        <ChapterMeta snapshot={snapshot} />
      </button>
    </h3>
  );
}
