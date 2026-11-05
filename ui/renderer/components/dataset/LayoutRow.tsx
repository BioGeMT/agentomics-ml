import type { LayoutNode } from "@/model/dataset";
import { FinderIcon } from "../symbols/FinderIcon";

/** A row of the layout list, then its children when open. */
export function LayoutRow({ node, depth, open, toggle }: { node: LayoutNode; depth: number; open: Set<string>; toggle: (name: string) => void }) {
  const folder = node.children !== undefined;
  const expandable = Boolean(node.children?.length);
  const expanded = open.has(node.name);
  return (
    <>
      <div className="finder-row" role="row" aria-expanded={expandable ? expanded : undefined}>
        <span className="finder-name" role="gridcell" style={{ paddingLeft: depth * 18 }}>
          {expandable ? (
            <button className="finder-disclosure" onClick={() => toggle(node.name)} aria-label={`${expanded ? "Collapse" : "Expand"} ${node.name}`}>
              <svg width="8" height="8" viewBox="0 0 8 8" aria-hidden className={expanded ? "open" : ""}><path d="M2 1l4 3-4 3z" fill="currentColor" /></svg>
            </button>
          ) : (
            <span className="finder-disclosure" />
          )}
          <FinderIcon folder={folder} />
          <span>{node.name}</span>
        </span>
        <span className="finder-note" role="gridcell">{node.note ?? (folder ? "Folder" : "")}</span>
      </div>
      {expandable && expanded && node.children!.map((child) => <LayoutRow key={child.name} node={child} depth={depth + 1} open={open} toggle={toggle} />)}
    </>
  );
}
