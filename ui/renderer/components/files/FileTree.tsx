import { useEffect } from "react";
import { useFolderListings } from "@/hooks/useFolderListings";
import { TreeFolder } from "./TreeFolder";

/**
 * The run's workspace as a folder tree: what the agent sees. Refreshes open folders while the run is live.
 * revealed: a folder to open, highlight and scroll to, e.g. one a step's result links to.
 */
export function FileTree({ root, live, selected, revealed, onOpen }: {
  root: string; live: boolean; selected: string | null; revealed?: string | null; onOpen: (rel: string) => void;
}) {
  const { listings, expanded, toggle, reveal } = useFolderListings(root, live);
  useEffect(() => { if (revealed) reveal(revealed); }, [revealed, reveal]);
  useEffect(() => {
    if (revealed && listings[revealed]) document.querySelector(`.file-tree [title="${CSS.escape(revealed)}"]`)?.scrollIntoView({ block: "center" });
  }, [revealed, listings]);
  return (
    <nav className="file-tree" aria-label="Run workspace">
      <p className="tree-root" title={root}>{root.split("/").pop()}</p>
      <ul><TreeFolder root={root} rel="" depth={0} listings={listings} expanded={expanded} selected={selected ?? revealed ?? null} onToggle={toggle} onOpen={onOpen} /></ul>
    </nav>
  );
}
