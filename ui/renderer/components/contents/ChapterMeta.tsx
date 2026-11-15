import type { Snapshot } from "@/model/run";

/** An iteration's validation score in the contents, tagged when it was the best. */
export function ChapterMeta({ snapshot }: { snapshot?: Snapshot }) {
  if (!snapshot) return null;
  const metric = snapshot.validation_metric ?? "AUROC";
  const v = snapshot.metrics[`validation/${metric}`];
  return (
    <span className="chapter-meta">
      {v !== undefined && `${metric} ${v.toFixed(3)}`}
      {snapshot.is_new_best && <span className="best-tag">best</span>}
    </span>
  );
}
