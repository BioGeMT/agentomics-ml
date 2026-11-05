import type { DatasetDetails } from "@agentomics/contract";
import { Check } from "lucide-react";
import { taskLabel } from "@/model/dataset";
import { ClassColumns } from "./ClassColumns";
import { DatasetInputs } from "./DatasetInputs";
import { ProportionBar } from "./ProportionBar";

/** A dataset that passed its check: the task, how rows split, how classes balance, and what the input is. */
export function DatasetSummary({ summary, details }: { summary: Record<string, any>; details: DatasetDetails }) {
  const train = Number(summary.splits?.train_rows ?? 0);
  const validation = summary.splits?.validation_rows != null ? Number(summary.splits.validation_rows) : null;
  const test = summary.test_rows != null ? Number(summary.test_rows) : null;
  const classes = details.classes ? Object.entries(details.classes).sort((a, b) => b[1] - a[1]) : [];
  const positive = summary.positive_class;
  const splits = [
    { name: "Train", value: train, tone: "train", note: validation == null ? "validation is split from it" : undefined },
    ...(validation != null ? [{ name: "Validation", value: validation, tone: "validation" }] : []),
    ...(test != null ? [{ name: "Test", value: test, tone: "test", note: "held out until the end" }] : []),
  ];
  return (
    <div className="ds-summary">
      <p className="dataset-status ok"><Check size={14} aria-hidden /> Ready to run · {taskLabel(summary, classes.length || Object.keys(summary.label_to_scalar ?? {}).length)}</p>
      <div className="ds-figures">
        <ProportionBar label="Splits" total={`${(train + (validation ?? 0) + (test ?? 0)).toLocaleString()} rows`} segments={splits} />
        {classes.length > 4 && <ClassColumns classes={classes} />}
        {classes.length > 0 && classes.length <= 4 && (
          <ProportionBar
            label="Classes in train"
            total={`${classes.length} classes`}
            segments={classes.map(([name, value]) => ({ name, value, tone: name === positive ? "positive" : "negative", note: name === positive ? "positive class" : undefined }))}
          />
        )}
      </div>
      <DatasetInputs summary={summary} features={details.features} />
    </div>
  );
}
