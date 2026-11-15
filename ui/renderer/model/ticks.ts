// Axis ticks for the charts.

/** Round ticks covering [low, high] with a step of 1, 2 or 5 × 10^n. */
export function niceTicks(low: number, high: number, count = 4): number[] {
  const span = Math.max(high - low, 1e-6);
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
  const start = Math.floor(low / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= high + step * 0.5; t += step) ticks.push(Number(t.toFixed(10)));
  return ticks;
}

export function decimals(ticks: number[]): number {
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 0.1;
  return Math.min(4, Math.max(1, -Math.floor(Math.log10(step))));
}
