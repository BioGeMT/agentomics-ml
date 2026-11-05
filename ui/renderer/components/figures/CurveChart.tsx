import { useChartWidth } from "@/hooks/useChartWidth";
import type { Point } from "@/model/evaluation";

export type Series = { key: string; label: string; points: Point[]; note?: string; tone: "selected" | "other" | "test" };

/** A curve on the unit square (ROC, precision–recall), one series per iteration plus the test split. */
export function CurveChart({ title, fig, xLabel, yLabel, series, diagonal }: { title: string; fig: string; xLabel: string; yLabel: string; series: Series[]; diagonal?: boolean }) {
  const [chartRef, width] = useChartWidth();
  const height = Math.round(Math.min(260, Math.max(190, width * 0.7)));
  const pad = { left: 34, right: 10, top: 10, bottom: 30 };
  const x = (v: number) => pad.left + v * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + (1 - v) * (height - pad.top - pad.bottom);
  const path = (points: Point[]) => points.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)} ${y(p.y).toFixed(1)}`).join(" ");
  const order = { other: 0, selected: 1, test: 2 } as const;
  return (
    <figure className="figure">
      <figcaption><span className="fig-label">{fig}</span> {title}</figcaption>
      <div className="chart" ref={chartRef}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={`${title}: ${series.map((s) => `${s.label} ${s.note ?? ""}`).join(", ")}`}>
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} className="grid-line" />
              <text x={pad.left - 6} y={y(t) + 3.5} className="axis-label" textAnchor="end">{t}</text>
              <text x={x(t)} y={height - 14} className="axis-label" textAnchor="middle">{t}</text>
            </g>
          ))}
          <text x={x(0.5)} y={height - 2} className="axis-title" textAnchor="middle">{xLabel}</text>
          <text x={10} y={y(0.5)} className="axis-title" textAnchor="middle" transform={`rotate(-90 10 ${y(0.5)})`}>{yLabel}</text>
          {diagonal && <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="chance-line" />}
          {[...series].sort((a, b) => order[a.tone] - order[b.tone]).map((s) => <path key={s.key} d={path(s.points)} className={`curve ${s.tone}`} />)}
        </svg>
      </div>
      <ul className="curve-legend">
        {series.map((s) => (
          <li key={s.key} className={s.tone}><i aria-hidden />{s.label}{s.note && <span>{s.note}</span>}</li>
        ))}
      </ul>
    </figure>
  );
}
