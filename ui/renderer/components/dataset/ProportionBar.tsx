export type Segment = { name: string; value: number; tone: string; note?: string };

/** A proportional bar with labelled segments; numbers sit in the legend so colour is never the only cue. */
export function ProportionBar({ label, total, segments }: { label: string; total: string; segments: Segment[] }) {
  const sum = segments.reduce((n, s) => n + s.value, 0) || 1;
  return (
    <figure className="ds-figure">
      <figcaption><span>{label}</span><span className="ds-total">{total}</span></figcaption>
      <div className="ds-bar" role="img" aria-label={segments.map((s) => `${s.name} ${s.value}`).join(", ")}>
        {segments.map((s) => <span key={s.name} className={`ds-seg ${s.tone}`} style={{ flexGrow: Math.max(s.value / sum, 0.02) }} />)}
      </div>
      <ul className="ds-legend">
        {segments.map((s) => (
          <li key={s.name}>
            <i className={`ds-swatch ${s.tone}`} aria-hidden />
            <span className="ds-name">{s.name}</span>
            <span className="ds-value">{s.value.toLocaleString()}</span>
            <span className="ds-share">{Math.round((s.value / sum) * 100)}%</span>
            {s.note && <span className="ds-note">{s.note}</span>}
          </li>
        ))}
      </ul>
    </figure>
  );
}
