import { useChartWidth } from "@/hooks/useChartWidth";
import { histogram, type Scored } from "@/model/evaluation";

/** Fig. 6: how predicted probabilities fall for each true class, with the 0.5 threshold. */
export function ScoreDistribution({ scored, names }: { scored: Scored; names: [string, string] }) {
  const [chartRef, width] = useChartWidth();
  const height = 150;
  const bins = 20;
  const { negative, positive } = histogram(scored, bins);
  const top = Math.max(...negative, ...positive, 1);
  const pad = { left: 10, right: 10, top: 8, bottom: 30 };
  const bw = (width - pad.left - pad.right) / bins;
  const h = (v: number) => (v / top) * (height - pad.top - pad.bottom);
  const base = height - pad.bottom;
  return (
    <figure className="figure">
      <figcaption><span className="fig-label">Fig. 6</span> Predicted probability by class</figcaption>
      <div className="chart" ref={chartRef}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={`Histogram of predicted probability for ${names[0]} and ${names[1]}`}>
          <line x1={pad.left} x2={width - pad.right} y1={base} y2={base} className="grid-line" />
          {negative.map((v, i) => v > 0 && <rect key={`n${i}`} x={pad.left + i * bw + 1} y={base - h(v)} width={bw / 2 - 1} height={h(v)} className="bar negative" />)}
          {positive.map((v, i) => v > 0 && <rect key={`p${i}`} x={pad.left + i * bw + bw / 2} y={base - h(v)} width={bw / 2 - 1} height={h(v)} className="bar positive" />)}
          <line x1={pad.left + (width - pad.left - pad.right) * 0.5} x2={pad.left + (width - pad.left - pad.right) * 0.5} y1={pad.top} y2={base} className="threshold-line" />
          {[0, 0.5, 1].map((t) => <text key={t} x={pad.left + t * (width - pad.left - pad.right)} y={height - 14} className="axis-label" textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"}>{t}</text>)}
          <text x={width / 2} y={height - 2} className="axis-title" textAnchor="middle">predicted probability of {names[1]}</text>
        </svg>
      </div>
      <ul className="curve-legend">
        <li className="neg"><i aria-hidden />{names[0]}</li>
        <li className="pos"><i aria-hidden />{names[1]}</li>
        <li className="thr"><i aria-hidden />threshold 0.5</li>
      </ul>
    </figure>
  );
}
