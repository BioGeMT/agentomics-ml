import { useState } from "react";
import { DATASET_LAYOUT } from "@/model/dataset";
import { LayoutRow } from "./LayoutRow";

/** The expected folder layout as a Finder list view, with folders that open and close. */
export function DatasetLayout() {
  const [open, setOpen] = useState<Set<string>>(new Set(["my_dataset", "train"]));
  const toggle = (name: string) => setOpen((current) => {
    const next = new Set(current);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    return next;
  });
  return (
    <div className="finder-list" role="treegrid" aria-label="Expected dataset folder layout">
      <div className="finder-head" role="row"><span role="columnheader">Name</span><span role="columnheader">Notes</span></div>
      <LayoutRow node={DATASET_LAYOUT} depth={0} open={open} toggle={toggle} />
    </div>
  );
}
