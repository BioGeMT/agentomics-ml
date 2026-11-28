import { BOUNDED, bestSnapshot, metricNames, sourceEntry, value } from "@/model/metrics";
import type { Snapshot } from "@/model/run";
import type { GoTo } from "./Figures";

/** every metric of the best iteration; choosing one plots it by iteration. */
export function MetricsTable({
  snapshots,
  primary,
  selected,
  onSelect,
  onGoTo,
}: {
  snapshots: Snapshot[];
  primary: string;
  selected: string;
  onSelect: (metric: string) => void;
  onGoTo: GoTo;
}) {
  const latest = snapshots[snapshots.length - 1];
  const best = bestSnapshot(snapshots);
  return (
    <figure className="figure">
      <figcaption>
        Metrics, iteration {best.iteration + 1}
        {best.is_new_best && <span className="best-tag">best</span>}
        <button className="entry-ref" onClick={() => onGoTo(sourceEntry(best))}>{sourceEntry(best)}</button>
      </figcaption>
      <table className="metrics-table">
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Validation</th>
            <th scope="col">Train</th>
          </tr>
        </thead>
        <tbody>
          {metricNames([best]).map((m) => {
            const plottable = BOUNDED.has(m);
            return (
              <tr key={m} className={[m === primary ? "primary" : "", m === selected ? "selected" : ""].join(" ")}>
                <th scope="row">
                  {plottable ? (
                    <button onClick={() => onSelect(m)} aria-pressed={m === selected} title={`Plot ${m} by iteration`}>{m}</button>
                  ) : (
                    m
                  )}
                </th>
                <td>{value(best, "validation", m)?.toFixed(4) ?? "—"}</td>
                <td>{value(best, "train", m)?.toFixed(4) ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="figure-note">
        Select a metric to plot it by iteration.
        {latest !== best && ` Latest iteration ${latest.iteration + 1} did not beat iteration ${best.iteration + 1}.`}
      </p>
    </figure>
  );
}
