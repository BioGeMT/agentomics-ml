import type { DirEntry } from "@agentomics/contract/protocol";
import { useCallback, useEffect, useState } from "react";
import { readWorkspace } from "@/model/workspace";

export type Listing = DirEntry[] | "loading" | { error: string };

/** The listings of a workspace's open folders, loaded as they open and refreshed every few seconds while live. */
export function useFolderListings(root: string, live: boolean) {
  const [listings, setListings] = useState<Record<string, Listing>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["", "run"]));

  const load = useCallback(async (rel: string, quiet = false) => {
    if (!quiet) setListings((current) => ({ ...current, [rel]: current[rel] && current[rel] !== "loading" ? current[rel] : "loading" }));
    const body = await readWorkspace({ root, path: rel }, "Couldn't read this folder.");
    const listing: Listing = "error" in body ? body : body.kind === "dir" ? body.entries : { error: "This is a file, not a folder." };
    setListings((current) => ({ ...current, [rel]: listing }));
  }, [root]);

  useEffect(() => { expanded.forEach((rel) => { if (!listings[rel]) load(rel); }); }, [expanded, listings, load]);
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => expanded.forEach((rel) => load(rel, true)), 5000);
    return () => clearInterval(timer);
  }, [live, expanded, load]);

  function toggle(rel: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(rel)) next.delete(rel);
      else next.add(rel);
      return next;
    });
  }

  return { listings, expanded, toggle };
}
