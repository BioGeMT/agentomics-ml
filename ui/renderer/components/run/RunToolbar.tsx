import { ChevronLeft, GitFork, Square } from "lucide-react";
import type { Side } from "@/hooks/usePanes";
import type { Run } from "@/model/run";
import { Logo } from "../brand/Logo";
import { PaneToggle } from "./PaneToggle";
import { RunCounts } from "./RunCounts";
import { RunStatus } from "./RunStatus";

/** The window's toolbar: sidebar toggles at each end, back to runs, the run's name, its status, Stop while it runs, Fork once an iteration has finished, and counts. */
export function RunToolbar({ title, model, run, elapsed, limit, offline, counts, shown, onToggle, onStop, forkHref }: {
  title: string;
  model?: string;
  run: Run;
  elapsed?: number;
  /** The run's time limit, in ms. */
  limit?: number;
  offline: boolean;
  counts: { done: number; tools: number };
  shown: Record<Side, boolean>;
  onToggle: (side: Side) => void;
  onStop?: () => void;
  /** New Run, set to fork this run from its last finished iteration (alongside Stop while it runs). */
  forkHref?: string;
}) {
  return (
    <header className="cover">
      {/* Three groups, over the page's three columns where the layout lines them up (notebook.css). */}
      <div className="cover-start">
        <PaneToggle side="contents" open={shown.contents} onToggle={() => onToggle("contents")} />
        <a href="/app/" className="cover-back" aria-label="Back to runs"><ChevronLeft size={18} strokeWidth={2} aria-hidden /></a>
        <Logo size={26} />
      </div>
      <div className="cover-middle">
        <div className="cover-run">
          <span className="cover-title">{title}</span>
          {model && <span className="cover-sub">{model}</span>}
        </div>
        <RunStatus run={run} elapsed={elapsed} limit={limit} offline={offline} />
        {onStop && (
          <button className="run-stop" onClick={onStop}>
            <Square size={10} fill="currentColor" strokeWidth={0} aria-hidden /> Stop
          </button>
        )}
        {forkHref && (
          <a className="run-stop run-fork" href={forkHref} title="Start a new run from this one: keep its iterations and add more">
            <GitFork size={12} strokeWidth={2.2} aria-hidden /> Fork
          </a>
        )}
      </div>
      <div className="cover-end">
        <RunCounts done={counts.done} tools={counts.tools} />
        <PaneToggle side="figures" open={shown.figures} onToggle={() => onToggle("figures")} />
      </div>
    </header>
  );
}
