/** Many classes: one column each in natural order, with a one-line verdict on balance. */
export function ClassColumns({ classes }: { classes: [string, number][] }) {
  const numeric = classes.every(([name]) => /^-?\d+(\.\d+)?$/.test(name));
  const ordered = [...classes].sort((a, b) => (numeric ? Number(a[0]) - Number(b[0]) : a[0].localeCompare(b[0])));
  const counts = classes.map(([, n]) => n);
  const min = Math.min(...counts), max = Math.max(...counts), total = counts.reduce((a, b) => a + b, 0);
  const balanced = min / max >= 0.8;
  const largest = classes.find(([, n]) => n === max)![0], smallest = classes.find(([, n]) => n === min)![0];
  return (
    <figure className="ds-figure">
      <figcaption><span>Classes in train</span><span className="ds-total">{classes.length} classes</span></figcaption>
      <p className="ds-verdict">
        {balanced ? <><b>Balanced</b> · {min === max ? `${min}` : `${min}–${max}`} per class</> : <><b>Imbalanced</b> · largest {largest} ({max}), smallest {smallest} ({min})</>}
      </p>
      <div className="ds-columns" role="img" aria-label={ordered.map(([name, n]) => `${name}: ${n}`).join(", ")}>
        {ordered.map(([name, n]) => (
          <div key={name} className="ds-column" title={`${name}: ${n.toLocaleString()} rows (${Math.round((n / total) * 100)}%)`}>
            <span className="ds-column-bar" style={{ height: `${Math.max((n / max) * 100, 3)}%` }} />
            <span className="ds-column-label">{name}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}
