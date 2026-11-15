import { useState } from "react";
import { useChartWidth } from "@/hooks/useChartWidth";
import { BOUNDED, sourceEntry, value } from "@/model/metrics";
import type { Snapshot } from "@/model/run";
import { decimals, niceTicks } from "@/model/ticks";
import type { GoTo } from "./Figures";
import { Legend } from "./Legend";
import { TRAIN, VALIDATION } from "./series";

/** Fig. 1: a metric for each iteration, validation and train, with the best marked. Click an iteration to open its evaluation. */
export function MetricByIteration({ snapshots, metric, primary, onGoTo }: { snapshots: Snapshot[]; metric: string; primary: string; onGoTo: GoTo }) {
  const [hover, setHover] = useState<number | null>(null);
  const [chartRef, width] = useChartWidth();
  const height = Math.round(Math.min(260, Math.max(176, width * 0.56)));
  const pad = { left: 40, right: 46, top: 22, bottom: 26 };
  const points = snapshots.map((s) => ({
    iteration: s.iteration + 1,
    entry: sourceEntry(s),
    validation: value(s, "validation", metric),
    train: value(s, "train", metric),
    best: s.is_new_best,
  }));
  const values = points.flatMap((p) => [p.validation, p.train]).filter((v): v is number => v !== undefined);
  const span = Math.max(Math.max(...values) - Math.min(...values), 0.01);
  let ticks = niceTicks(Math.min(...values) - span * 0.25, Math.max(...values) + span * 0.25);
  if (BOUNDED.has(metric)) ticks = ticks.filter((t) => t <= 1 && t >= -1);
  const low = ticks[0];
  const high = ticks[ticks.length - 1];
  const digits = decimals(ticks);
  const n = points.length;
  const plotWidth = width - pad.left - pad.right;
  // Points sit inset from the y-axis so marks and labels never meet the tick column.
  const inset = 16;
  const x = (i: number) => (n === 1 ? pad.left + plotWidth / 2 : pad.left + inset + ((plotWidth - inset * 2) * i) / (n - 1));
  const y = (v: number) => pad.top + (height - pad.top - pad.bottom) * (1 - (v - low) / (high - low || 1));
  const line = (key: "validation" | "train") =>
    points.map((p, i) => (p[key] === undefined ? "" : `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(p[key]!).toFixed(1)}`)).join(" ");
  const last = points[n - 1];
  const bestIndex = points.map((p) => p.best).lastIndexOf(true);
  const best = bestIndex >= 0 ? points[bestIndex] : undefined;
  const column = n === 1 ? 96 : plotWidth / (n - 1);

  return (
    <figure className="figure">
      <figcaption>
        <span className="fig-label">Fig. 1</span> {metric} by iteration
        {metric !== primary && <span className="fig-aside">run optimises {primary}</span>}
      </figcaption>
      <Legend />
      <div className="chart" ref={chartRef} onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={`Validation and train ${metric} for each iteration`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={pad.left} x2={width - pad.right} y1={y(t)} y2={y(t)} className="grid-line" />
              <text x={pad.left - 7} y={y(t) + 3.5} className="axis-label" textAnchor="end">{t.toFixed(digits)}</text>
            </g>
          ))}
          {points.map((p, i) => (
            <text key={p.iteration} x={x(i)} y={height - 8} className="axis-label" textAnchor="middle">{p.iteration}</text>
          ))}
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={pad.top} y2={height - pad.bottom} className="crosshair" />}
          <path d={line("train")} fill="none" stroke={TRAIN} strokeWidth="2" strokeDasharray="5 4" strokeLinejoin="round" />
          <path d={line("validation")} fill="none" stroke={VALIDATION} strokeWidth="2" strokeLinejoin="round" />
          {points.map((p, i) => (
            <g key={p.iteration}>
              {p.train !== undefined && <circle cx={x(i)} cy={y(p.train)} r="3.6" fill="var(--page)" stroke={TRAIN} strokeWidth="2" />}
              {p.validation !== undefined && <circle cx={x(i)} cy={y(p.validation)} r="4.5" fill={VALIDATION} stroke="var(--page)" strokeWidth="2" />}
            </g>
          ))}
          {best?.validation !== undefined && (
            <circle cx={x(bestIndex)} cy={y(best.validation)} r="8.5" fill="none" stroke={VALIDATION} strokeWidth="1.5" />
          )}
          {last.validation !== undefined && (
            <text x={x(n - 1) + 13} y={y(last.validation) + 4} className="direct-label">
              {last.validation.toFixed(3)}
              {bestIndex === n - 1 && <tspan className="best-label"> best</tspan>}
            </text>
          )}
          {best?.validation !== undefined && bestIndex !== n - 1 && (
            <text x={x(bestIndex) - 4} y={y(best.validation) + 22} className="best-label">best · {best.validation.toFixed(3)}</text>
          )}
          {points.map((p, i) => (
            <rect
              key={`hit-${p.iteration}`}
              x={x(i) - column / 2}
              y={pad.top}
              width={column}
              height={height - pad.top - pad.bottom}
              fill="transparent"
              className="hit"
              onMouseEnter={() => setHover(i)}
              onClick={() => onGoTo(p.entry)}
            >
              <title>{`Iteration ${p.iteration}: open entry ${p.entry}`}</title>
            </rect>
          ))}
        </svg>
        {hover !== null && (
          <div className="tooltip" style={{ left: `${(x(hover) / width) * 100}%` }}>
            <strong>Iteration {points[hover].iteration}{points[hover].best ? " · best" : ""}</strong>
            <span><i className="swatch-validation" />Validation {points[hover].validation?.toFixed(4) ?? "—"}</span>
            <span><i className="swatch-train" />Train {points[hover].train?.toFixed(4) ?? "—"}</span>
            <em>From entry {points[hover].entry} · click to open</em>
          </div>
        )}
      </div>
      {n === 1 && (
        <p className="figure-note">
          {(snapshots[0]?.iterations_total ?? 1) > 1 ? "One iteration so far; the line forms as more finish." : "This run had a single iteration."}
        </p>
      )}
    </figure>
  );
}
