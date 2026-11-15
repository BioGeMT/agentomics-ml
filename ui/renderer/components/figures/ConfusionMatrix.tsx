import { confusion, type Scored } from "@/model/evaluation";

/** Fig. 5: true against predicted class at the 0.5 threshold, shaded by share of each true class. */
export function ConfusionMatrix({ scored, names }: { scored: Scored; names: [string, string] }) {
  const m = confusion(scored);
  const rows: [string, number, number][] = [[names[0], m.tn, m.fp], [names[1], m.fn, m.tp]];
  return (
    <figure className="figure">
      <figcaption><span className="fig-label">Fig. 5</span> Confusion at 0.5</figcaption>
      <table className="confusion" aria-label="Confusion matrix: rows are true classes, columns predicted classes">
        <thead><tr><th /><th colSpan={2}>Predicted</th></tr><tr><th>True</th><th>{names[0]}</th><th>{names[1]}</th></tr></thead>
        <tbody>
          {rows.map(([name, a, b], r) => {
            const total = a + b || 1;
            return (
              <tr key={name}>
                <th>{name}</th>
                {[a, b].map((v, c) => (
                  <td key={c} className={r === c ? "hit" : "miss"} style={{ ["--share" as string]: v / total }}>
                    <b>{v}</b><span>{Math.round((v / total) * 100)}%</span>
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </figure>
  );
}
