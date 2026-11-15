import type { TestResult } from "@/model/evaluation";
import { METRIC_ORDER } from "@/model/metrics";
import type { Snapshot } from "@/model/run";

/** The number the run is for: the best model on data it never saw. */
export function TestResultCard({ test, best, primary }: { test: TestResult; best?: Snapshot; primary: string }) {
  const metrics = METRIC_ORDER.filter((m) => m in test.metrics);
  const headline = primary in test.metrics ? primary : metrics[0];
  return (
    <figure className="figure test-result">
      <figcaption><span className="fig-label">Result</span> Held-out test{best && <span className="fig-aside">best model, iteration {best.iteration + 1}</span>}</figcaption>
      <p className="test-headline"><span>{headline}</span><b>{test.metrics[headline]?.toFixed(4)}</b></p>
      <dl className="test-metrics">
        {metrics.filter((m) => m !== headline).map((m) => (
          <div key={m}><dt>{m}</dt><dd>{test.metrics[m].toFixed(4)}</dd></div>
        ))}
      </dl>
    </figure>
  );
}
