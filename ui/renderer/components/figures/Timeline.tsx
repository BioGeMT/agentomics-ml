// Where a run spends its time: one bar per iteration, a segment per stage on a shared scale,
// then time and cost per stage summed over the run.
import { formatDuration } from "@/model/format";
import { addUsage, STEP_SEQUENCE, type Entry, type Usage } from "@/model/run";
import { formatCost, usageTitle } from "@/model/usage";

// Stages run in a fixed order, so they share one hue stepped light to dark; a running stage is amber.
const RAMP = ["#c6d5f6", "#aec4f2", "#94b1ec", "#7a9de6", "#628ade", "#4b77d5", "#3a63c2", "#2f52a8", "#26438b"];

function title(stage: string) {
  return stage.charAt(0) + stage.slice(1).toLowerCase();
}

export function Timeline({ entries, now, onGoTo }: { entries: Entry[]; now: number; onGoTo: (key: string) => void }) {
  const steps = entries.filter((e) => e.position > 0 && e.startedAt && (e.status === "done" || e.status === "running" || e.status === "failed"));
  if (steps.length === 0) return null;
  const length = (e: Entry) => Math.max(0, (e.endedAt ?? now) - (e.startedAt ?? now));
  const iterations = [...new Set(steps.map((e) => e.iteration))].sort((a, b) => a - b);
  const totals = iterations.map((it) => steps.filter((e) => e.iteration === it).reduce((sum, e) => sum + length(e), 0));
  const longest = Math.max(...totals, 1);
  const byStage = STEP_SEQUENCE.map((stage, index) => {
    const own = steps.filter((e) => e.position === index + 1);
    return {
      stage: title(stage),
      index,
      time: own.reduce((sum, e) => sum + length(e), 0),
      usage: own.reduce<Usage | undefined>((sum, e) => addUsage(sum, e.usage), undefined),
    };
  }).filter((s) => s.time > 0);
  const runTotal = totals.reduce((a, b) => a + b, 0) || 1;
  // Runs from before usage was recorded have no cost; leave the column out rather than show dashes.
  const hasCost = byStage.some((s) => s.usage);

  return (
    <figure className="figure timeline">
      <figcaption><span className="fig-label">Fig. 7</span> Time by stage</figcaption>
      <div className="timeline-rows">
        {iterations.map((it, row) => (
          <div key={it} className="timeline-row">
            <span className="timeline-label">{it}</span>
            <div className="timeline-track">
              <div className="timeline-bar" style={{ width: `${(totals[row] / longest) * 100}%` }}>
                {steps.filter((e) => e.iteration === it).map((e) => (
                  <button
                    key={e.key}
                    className={`timeline-seg ${e.status}`}
                    style={{ flexGrow: Math.max(length(e), 1), background: e.status === "running" ? undefined : RAMP[e.position - 1] }}
                    title={`${e.key} ${e.title} · ${formatDuration(length(e))}${e.usage ? ` · ${formatCost(e.usage)}` : ""}`}
                    aria-label={`${e.key} ${e.title}, ${formatDuration(length(e))}`}
                    onClick={() => onGoTo(e.key)}
                  />
                ))}
              </div>
            </div>
            <span className="timeline-total">{formatDuration(totals[row])}</span>
          </div>
        ))}
      </div>
      <table className="metrics-table stage-table">
        <thead><tr><th>Stage</th><th>Time</th>{hasCost && <th>Cost</th>}</tr></thead>
        <tbody>
          {byStage.map((s) => (
            <tr key={s.stage}>
              <th><i className="stage-swatch" style={{ background: RAMP[s.index] }} aria-hidden />{s.stage}</th>
              <td>{formatDuration(s.time)} <span className="stage-share">{Math.round((s.time / runTotal) * 100)}%</span></td>
              {hasCost && <td title={usageTitle(s.usage)}>{formatCost(s.usage) ?? "—"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
