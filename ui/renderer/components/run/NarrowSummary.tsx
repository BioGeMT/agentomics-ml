import { formatDuration } from "@/model/format";
import type { Entry, Run } from "@/model/run";

/** Narrow windows lead with the answer: best metric, its trend, and what the agent is doing now. */
export function NarrowSummary({ run, live, now }: { run: Run; live?: Entry; now: number }) {
  const latest = run.snapshots[run.snapshots.length - 1];
  const metric = latest?.validation_metric ?? "AUROC";
  const best = [...run.snapshots].reverse().find((s) => s.is_new_best) ?? latest;
  const values = run.snapshots.map((s) => s.metrics[`validation/${metric}`]).filter((v): v is number => v !== undefined);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const point = (v: number, i: number): [number, number] => [
    values.length === 1 ? 54 : 8 + (i * 92) / (values.length - 1),
    8 + 26 * (1 - (v - low) / (high - low || 1)),
  ];
  return (
    <div className="narrow-summary" aria-label="Run summary">
      <div>
        <small>{best ? `Best so far · iteration ${best.iteration + 1} · ${metric}` : "No iteration evaluated yet"}</small>
        <span className="big">{best ? best.metrics[`validation/${metric}`]?.toFixed(4) : "—"}</span>
      </div>
      {values.length > 0 && (
        <svg width="108" height="42" viewBox="0 0 108 42" role="img" aria-label={`Validation ${metric} by iteration: ${values.map((v) => v.toFixed(3)).join(", ")}`}>
          {values.length > 1 && <path d={`M${values.map((v, i) => point(v, i).join(" ")).join(" L")}`} fill="none" stroke="var(--series-validation)" strokeWidth="2" />}
          {values.map((v, i) => {
            const [cx, cy] = point(v, i);
            return <circle key={i} cx={cx} cy={cy} r="3.5" fill="var(--series-validation)" />;
          })}
        </svg>
      )}
      {live && (
        <small className="narrow-now">
          Now: {live.key} {live.title}{live.startedAt ? ` · ${formatDuration(now - live.startedAt)}` : ""}
        </small>
      )}
    </div>
  );
}
