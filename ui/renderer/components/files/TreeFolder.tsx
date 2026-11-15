import { LoaderCircle } from "lucide-react";
import type { Listing } from "@/hooks/useFolderListings";
import { DownloadLink } from "./DownloadLink";
import { TreeRow } from "./TreeRow";

/** The contents of an open folder: its rows, each folder opening into its own contents. */
export function TreeFolder({ root, rel, depth, listings, expanded, selected, onToggle, onOpen }: {
  root: string;
  rel: string;
  depth: number;
  listings: Record<string, Listing>;
  expanded: Set<string>;
  selected: string | null;
  onToggle: (rel: string) => void;
  onOpen: (rel: string) => void;
}) {
  const listing = listings[rel];
  const indent = { paddingLeft: 12 + depth * 14 };
  if (!listing || listing === "loading") return <li className="tree-note" style={indent}><LoaderCircle size={12} className="spin" aria-hidden /> Loading…</li>;
  if ("error" in listing) return <li className="tree-note error" style={indent}>{listing.error}</li>;
  if (listing.length === 0) return <li className="tree-note" style={indent}>Empty</li>;
  return listing.map((entry) => {
    const path = rel ? `${rel}/${entry.name}` : entry.name;
    const open = expanded.has(path);
    return (
      <li key={path}>
        <TreeRow entry={entry} path={path} depth={depth} open={open} selected={selected === path} onClick={() => (entry.dir ? onToggle(path) : onOpen(path))} />
        <DownloadLink root={root} rel={path} folder={entry.dir} className="tree-download" />
        {entry.dir && open && (
          <ul><TreeFolder root={root} rel={path} depth={depth + 1} listings={listings} expanded={expanded} selected={selected} onToggle={onToggle} onOpen={onOpen} /></ul>
        )}
      </li>
    );
  });
}
