import { useState } from "react";
import { useChartWidth } from "@/hooks/useChartWidth";
import { BOUNDED, metricNames, sourceEntry, value } from "@/model/metrics";
import type { Snapshot } from "@/model/run";
import { decimals, niceTicks } from "@/model/ticks";
import type { GoTo } from "./Figures";
import { Legend } from "./Legend";
import { TRAIN, VALIDATION } from "./series";

/** Fig. 2: train against validation for each metric of one iteration, as dumbbells; the gap shows overfitting. */
export function TrainVsValidation({ snapshot, primary, onGoTo }: { snapshot: Snapshot; primary: string; onGoTo: GoTo }) {
  const [hover, setHover] = useState<string | null>(null);
  const metrics = metricNames([snapshot]).filter((m) => BOUNDED.has(m));
  const [chartRef, width] = useChartWidth();
  const row = 26;
  const pad = { left: 58, right: 16, top: 8, bottom: 22 };
  const height = pad.top + metrics.length * row + pad.bottom;
  const all = metrics.flatMap((m) => [value(snapshot, "validation", m), value(snapshot, "train", m)]).filter((v): v is number => v !== undefined);
  const span = Math.max(Math.max(...all) - Math.min(...all), 0.01);
  const ticks = niceTicks(Math.min(...all) - span * 0.15, Math.min(1, Math.max(...all) + span * 0.1), 3).filter((t) => t <= 1);
  const low = ticks[0];
  const high = Math.max(ticks[ticks.length - 1], Math.max(...all));
  const digits = decimals(ticks);
  const x = (v: number) => pad.left + (width - pad.left - pad.right) * ((v - low) / (high - low || 1));

  return (
    <figure className="figure">
      <figcaption>
        <span className="fig-label">Fig. 2</span> Train vs validation, iteration {snapshot.iteration + 1}
        <button className="entry-ref" onClick={() => onGoTo(sourceEntry(snapshot))}>{sourceEntry(snapshot)}</button>
      </figcaption>
      <Legend />
      <div className="chart" ref={chartRef} onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={`Train and validation metrics for iteration ${snapshot.iteration + 1}`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={pad.top} y2={height - pad.bottom} className="grid-line" />
              <text x={x(t)} y={height - 7} className="axis-label" textAnchor="middle">{t.toFixed(digits)}</text>
            </g>
          ))}
          {metrics.map((m, i) => {
            const cy = pad.top + i * row + row / 2;
            const v = value(snapshot, "validation", m);
            const t = value(snapshot, "train", m);
            return (
              <g key={m} onMouseEnter={() => setHover(m)}>
                <rect x={0} y={cy - row / 2} width={width} height={row} fill={hover === m ? "var(--hover-row)" : "transparent"} />
                <text x={pad.left - 8} y={cy + 4} textAnchor="end" className={m === primary ? "axis-label primary" : "axis-label"}>{m}</text>
                {v !== undefined && t !== undefined && <line x1={x(Math.min(v, t))} x2={x(Math.max(v, t))} y1={cy} y2={cy} className="dumbbell" />}
                {t !== undefined && <circle cx={x(t)} cy={cy} r="3.6" fill="var(--page)" stroke={TRAIN} strokeWidth="2" />}
                {v !== undefined && <circle cx={x(v)} cy={cy} r="4.5" fill={VALIDATION} stroke="var(--page)" strokeWidth="2" />}
              </g>
            );
          })}
        </svg>
        {hover && (
          <div className="tooltip tooltip-row" style={{ top: `${((pad.top + metrics.indexOf(hover) * row) / height) * 100}%` }}>
            <strong>{hover}</strong>
            <span><i className="swatch-validation" />Validation {value(snapshot, "validation", hover)?.toFixed(4)}</span>
            <span><i className="swatch-train" />Train {value(snapshot, "train", hover)?.toFixed(4)}</span>
            <em>Gap {Math.abs((value(snapshot, "train", hover) ?? 0) - (value(snapshot, "validation", hover) ?? 0)).toFixed(4)}</em>
          </div>
        )}
      </div>
    </figure>
  );
}
