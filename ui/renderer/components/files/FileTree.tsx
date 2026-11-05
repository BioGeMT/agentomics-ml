import { useFolderListings } from "@/hooks/useFolderListings";
import { TreeFolder } from "./TreeFolder";

/** The run's workspace as a folder tree: what the agent sees. Refreshes open folders while the run is live. */
export function FileTree({ root, live, selected, onOpen }: { root: string; live: boolean; selected: string | null; onOpen: (rel: string) => void }) {
  const { listings, expanded, toggle } = useFolderListings(root, live);
  return (
    <nav className="file-tree" aria-label="Run workspace">
      <p className="tree-root" title={root}>{root.split("/").pop()}</p>
      <ul><TreeFolder root={root} rel="" depth={0} listings={listings} expanded={expanded} selected={selected} onToggle={toggle} onOpen={onOpen} /></ul>
    </nav>
  );
}
