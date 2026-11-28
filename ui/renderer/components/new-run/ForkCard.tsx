import type { RunEntry } from "@agentomics/contract";
import { GitFork } from "lucide-react";

/**
 * The run a fork starts from: what it reached, and the iteration to fork from. The
 * fork keeps iterations 1..kept with their files and data split, and adds its own iterations on top.
 */
export function ForkCard({ source, kept, onKept }: { source: RunEntry; kept: number; onKept: (n: number) => void }) {
  const done = source.summary?.done ?? 0;
  const { metric, best } = source.summary ?? {};
  return (
    <section className="dataset-card fork-card">
      <div className="dataset-head">
        <span className="dataset-icon" aria-hidden><GitFork size={24} strokeWidth={1.8} /></span>
        <div>
          <strong>Fork {source.dataset}</strong>
          <span className="dataset-where">{source.id} · {source.model}</span>
        </div>
        <a className="mac-secondary fork-cancel" href="/app/">Cancel</a>
      </div>
      <p className="fork-facts">
        {done} iteration{done === 1 ? "" : "s"} finished{metric && best != null ? ` · best ${metric} ${best.toFixed(3)}` : ""}
      </p>
      {done > 0 ? (
        <>
          <label className="form-row fork-keep">
            <span>Fork from</span>
            <select value={kept} onChange={(e) => onKept(Number(e.target.value))}>
              {Array.from({ length: done }, (_, i) => i + 1).map((n) => <option key={n} value={n}>Iteration {n}</option>)}
            </select>
          </label>
          <p className="fork-note">
            The fork keeps {kept > 1 ? `iterations 1–${kept}` : "iteration 1"}, with {kept > 1 ? "their" : "its"} files and data split. The run's goal stays;
            what you write under What to change is added to it for the new iterations.
          </p>
        </>
      ) : (
        <p className="dataset-status error">This run has no finished iteration to fork from.</p>
      )}
    </section>
  );
}
