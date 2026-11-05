// What the lobby knows about datasets: the documented folder layout and how to name a task.

export type LayoutNode = { name: string; note?: string; children?: LayoutNode[] };

/** The documented dataset layout (docs/user-guide/datasets.md), so the first screen says what to bring. */
export const DATASET_LAYOUT: LayoutNode = {
  name: "my_dataset",
  note: "the folder you choose",
  children: [
    { name: "train", note: "required", children: [{ name: "input", note: "data files, any format", children: [] }, { name: "labels.csv", note: "id,label" }] },
    { name: "validation", note: "optional", children: [] },
    { name: "test", note: "optional · held out, scores the best model", children: [] },
    { name: "metadata.json", note: "optional" },
    { name: "dataset_description.md", note: "optional · context for the agent" },
  ],
};

export function taskLabel(summary: Record<string, any>, classes: number): string {
  if (summary.task_type === "regression") return "Regression";
  if (summary.task_type === "classification") return classes === 2 ? "Binary classification" : classes > 2 ? `${classes}-class classification` : "Classification";
  return String(summary.task_type ?? "Unknown task");
}
