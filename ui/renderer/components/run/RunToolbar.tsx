import { ChevronLeft } from "lucide-react";
import type { Side } from "@/hooks/usePanes";
import type { Run } from "@/model/run";
import { Logo } from "../brand/Logo";
import { PaneToggle } from "./PaneToggle";
import { RunCounts } from "./RunCounts";
import { RunStatus } from "./RunStatus";

/** The window's toolbar: sidebar toggles at each end, back to runs, the run's name, its status and counts. */
export function RunToolbar({ title, model, run, elapsed, offline, counts, shown, onToggle }: {
  title: string;
  model?: string;
  run: Run;
  elapsed?: number;
  offline: boolean;
  counts: { done: number; tools: number };
  shown: Record<Side, boolean>;
  onToggle: (side: Side) => void;
}) {
  return (
    <header className="cover">
      <PaneToggle side="contents" open={shown.contents} onToggle={() => onToggle("contents")} />
      <a href="/app/" className="cover-back" aria-label="Back to runs"><ChevronLeft size={18} strokeWidth={2} aria-hidden /></a>
      <Logo size={26} />
      <div className="cover-run">
        <span className="cover-title">{title}</span>
        {model && <span className="cover-sub">{model}</span>}
      </div>
      <RunStatus run={run} elapsed={elapsed} offline={offline} />
      <RunCounts done={counts.done} tools={counts.tools} />
      <PaneToggle side="figures" open={shown.figures} onToggle={() => onToggle("figures")} />
    </header>
  );
}
