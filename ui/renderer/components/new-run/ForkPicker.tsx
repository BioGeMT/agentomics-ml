import type { RunEntry } from "@agentomics/contract";
import { GitFork } from "lucide-react";
import { dayLabel } from "@/model/run-list";
import { forkLink } from "@/model/run-link";

const SHOWN = 5;

/** New Run's other start: fork one of the recent runs that has a scored iteration, instead of a new dataset. */
export function ForkPicker({ runs }: { runs: RunEntry[] }) {
  const forkable = runs.filter((r) => (r.summary?.done ?? 0) > 0 && r.summary?.best != null);
  if (forkable.length === 0) return null;
  return (
    <section className="dataset-card fork-picker" aria-labelledby="fork-picker-title">
      <h2 id="fork-picker-title">Fork a Run</h2>
      <p className="dataset-empty-lead">Keep a run's iterations, with their files and data split, and try something different from there.</p>
      <ul>
        {forkable.slice(0, SHOWN).map((run) => {
          const { done = 0, metric, best } = run.summary ?? {};
          return (
            <li key={run.id}>
              <div>
                <strong>{run.dataset}</strong>
                <span>
                  {done} iteration{done === 1 ? "" : "s"}{metric && best != null ? ` · best ${metric} ${best.toFixed(3)}` : ""}
                  {" · "}{dayLabel(run.startedAt)} {new Date(run.startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · {run.model}
                </span>
              </div>
              <a className="mac-secondary" href={forkLink(run.workspace)}>
                <GitFork size={12} strokeWidth={2.2} aria-hidden /> Fork
              </a>
            </li>
          );
        })}
      </ul>
      {forkable.length > SHOWN && <p className="fork-picker-more">Older runs: open one from the list and use its Fork button.</p>}
    </section>
  );
}
