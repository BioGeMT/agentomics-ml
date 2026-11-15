import { formatDuration } from "@/model/format";
import type { Entry, Snapshot } from "@/model/run";
import { formatCost, iterationUsage, usageTitle } from "@/model/usage";
import { Disclosure } from "./Disclosure";

/** Notebook: the chapter break opens each iteration; folded, it summarises what is hidden. */
export function ChapterBreak({ iteration, snapshot, entries, open, onToggle }: {
  iteration: number;
  snapshot?: Snapshot;
  entries: Entry[];
  open: boolean;
  onToggle: (event: React.MouseEvent) => void;
}) {
  const metric = snapshot?.validation_metric ?? "AUROC";
  const v = snapshot?.metrics[`validation/${metric}`];
  const steps = entries.filter((e) => e.position > 0 && e.status !== "skipped" && e.status !== "upcoming");
  const started = Math.min(...entries.map((e) => e.startedAt ?? Infinity));
  const ended = Math.max(...entries.map((e) => e.endedAt ?? e.startedAt ?? -Infinity));
  const usage = iterationUsage(entries, iteration);
  return (
    <div className={`chapter-break ${open ? "" : "folded"}`}>
      <div className="margin" />
      <div className="chapter-break-body">
        <h2>
          <button className="chapter-toggle" aria-expanded={open} onClick={onToggle} title="Option-click to fold or unfold every iteration">
            <Disclosure open={open} />
            Iteration {iteration}
          </button>
        </h2>
        {v !== undefined && <span className="chapter-result">Validation {metric} {v.toFixed(4)}</span>}
        {snapshot?.is_new_best && <span className="best-tag">best so far</span>}
        {snapshot && !snapshot.is_new_best && <span className="chapter-result muted">did not beat the best</span>}
        {usage && <span className="chapter-cost" title={usageTitle(usage)}>{formatCost(usage)}</span>}
        {!open && (
          <span className="chapter-folded">
            {steps.length} {steps.length === 1 ? "entry" : "entries"} folded
            {Number.isFinite(started) && Number.isFinite(ended) && ` · ${formatDuration(ended - started)}`}
          </span>
        )}
      </div>
    </div>
  );
}
