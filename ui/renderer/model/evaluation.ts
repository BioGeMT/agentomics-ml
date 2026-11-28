// Loads each iteration's validation predictions and the held-out test result from the run's workspace,
// and computes the curves the Figures pane draws. Binary classification only (probability_1 column).

import { readText } from "./workspace";

export type Scored = { label: 0 | 1; score: number }[];
export type IterationEval = { iteration: number; scored: Scored }; // iteration is 1-based
export type TestResult = { metrics: Record<string, number>; scored?: Scored };

const read = readText;

function table(text: string): Record<string, string>[] {
  const [head, ...rows] = text.trim().split(/\r?\n/);
  const columns = head.split(",");
  return rows.map((row) => {
    const cells = row.split(",");
    return Object.fromEntries(columns.map((c, i) => [c, cells[i]]));
  });
}

/** Joins predictions (id, probability_1) to labels (id, numeric_label). Null unless the task is binary. */
function join(predictions: string, labels: string): Scored | null {
  const preds = table(predictions);
  if (!preds.length || !("probability_1" in preds[0])) return null;
  const truth = new Map(table(labels).map((r) => [r.id, Number(r.numeric_label ?? r.label)]));
  const scored: Scored = [];
  for (const p of preds) {
    const label = truth.get(p.id);
    const score = Number(p.probability_1);
    if ((label === 0 || label === 1) && Number.isFinite(score)) scored.push({ label, score });
  }
  return scored.length ? scored : null;
}

const toRel = (containerPath: string) => containerPath.replace(/^\/workspace\/?/, "").replace(/\/$/, "");

/** Validation predictions for each finished iteration, joined to the labels of the split it used. */
export async function loadIterations(root: string, count: number): Promise<IterationEval[]> {
  const results: IterationEval[] = [];
  let validationDir: string | null = null;
  for (let i = 0; i < count; i++) {
    const split = await read(root, `run/iteration_${i}/data_split/output.json`);
    if (split) {
      try {
        const path = JSON.parse(split).payload?.val_path;
        if (typeof path === "string" && path.startsWith("/workspace/run/")) validationDir = toRel(path);
      } catch {}
    }
    if (!validationDir) continue; // splitting skipped before any split was recorded
    const [predictions, labels] = await Promise.all([
      read(root, `run/iteration_${i}/validation_evaluation/eval_predictions_validation.csv`),
      read(root, `${validationDir}/labels.csv`),
    ]);
    const scored = predictions && labels ? join(predictions, labels) : null;
    if (scored) results.push({ iteration: i + 1, scored });
  }
  return results;
}

/** The best iteration's score on the held-out test split, when the run has evaluated it. */
export async function loadTest(root: string): Promise<TestResult | null> {
  const metrics = await read(root, "best_iteration_snapshot/eval_predictions_test.metrics.json");
  if (!metrics) return null;
  let parsed: Record<string, number>;
  try {
    parsed = JSON.parse(metrics);
  } catch {
    return null;
  }
  const [predictions, labels] = await Promise.all([
    read(root, "best_iteration_snapshot/eval_predictions_test.csv"),
    read(root, "best_iteration_snapshot/eval_predictions_test.numeric_labels.csv"),
  ]);
  return { metrics: parsed, scored: predictions && labels ? join(predictions, labels) ?? undefined : undefined };
}

export type Point = { x: number; y: number };

/** ROC curve (false positive rate, true positive rate), sweeping the threshold over every distinct score. */
export function roc(scored: Scored): { points: Point[]; auc: number } {
  const sorted = [...scored].sort((a, b) => b.score - a.score);
  const positives = sorted.filter((s) => s.label === 1).length || 1;
  const negatives = sorted.length - positives || 1;
  const points: Point[] = [{ x: 0, y: 0 }];
  let tp = 0, fp = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].label === 1) tp++;
    else fp++;
    if (i === sorted.length - 1 || sorted[i + 1].score !== sorted[i].score) points.push({ x: fp / negatives, y: tp / positives });
  }
  let auc = 0;
  for (let i = 1; i < points.length; i++) auc += (points[i].x - points[i - 1].x) * (points[i].y + points[i - 1].y) / 2;
  return { points, auc };
}

/** Precision-recall curve (recall, precision). */
export function precisionRecall(scored: Scored): Point[] {
  const sorted = [...scored].sort((a, b) => b.score - a.score);
  const positives = sorted.filter((s) => s.label === 1).length || 1;
  const points: Point[] = [];
  let tp = 0, fp = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].label === 1) tp++;
    else fp++;
    if (i === sorted.length - 1 || sorted[i + 1].score !== sorted[i].score) points.push({ x: tp / positives, y: tp / (tp + fp) });
  }
  return [{ x: 0, y: points[0]?.y ?? 1 }, ...points];
}

export function confusion(scored: Scored, threshold = 0.5) {
  const m = { tn: 0, fp: 0, fn: 0, tp: 0 };
  for (const s of scored) {
    const predicted = s.score >= threshold ? 1 : 0;
    if (s.label === 1) predicted ? m.tp++ : m.fn++;
    else predicted ? m.fp++ : m.tn++;
  }
  return m;
}

/** Counts of scores per class in equal-width bins over [0, 1]. */
export function histogram(scored: Scored, bins = 20) {
  const negative = new Array(bins).fill(0);
  const positive = new Array(bins).fill(0);
  for (const s of scored) {
    const bin = Math.min(bins - 1, Math.max(0, Math.floor(s.score * bins)));
    (s.label === 1 ? positive : negative)[bin]++;
  }
  return { negative, positive };
}

/** [negative, positive] class names from the run's dataset metadata; generic names when absent. */
export async function loadClassNames(root: string): Promise<[string, string]> {
  const text = await read(root, "run/shared/dataset_metadata.json");
  try {
    const meta = text ? JSON.parse(text) : {};
    if (meta.negative_class && meta.positive_class) return [String(meta.negative_class), String(meta.positive_class)];
  } catch {}
  return ["negative", "positive"];
}
