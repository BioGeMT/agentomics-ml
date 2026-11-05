import type { DirEntry } from "@agentomics/contract/protocol";
import { ChevronRight, Folder } from "lucide-react";
import { bytes } from "@/model/format";
import { FileIcon } from "./FileIcon";

/** One file or folder in the tree. */
export function TreeRow({ entry, path, depth, open, selected, onClick }: {
  entry: DirEntry; path: string; depth: number; open: boolean; selected: boolean; onClick: () => void;
}) {
  return (
    <button
      className={`tree-row ${selected ? "selected" : ""}`}
      style={{ paddingLeft: 8 + depth * 14 }}
      onClick={onClick}
      aria-expanded={entry.dir ? open : undefined}
      title={path}
    >
      <span className="tree-disclosure">{entry.dir && <ChevronRight size={12} className={`disclosure ${open ? "open" : ""}`} aria-hidden />}</span>
      <span className="tree-icon">{entry.dir ? <Folder size={14} aria-hidden /> : <FileIcon name={entry.name} />}</span>
      <span className="tree-name">{entry.name}</span>
      {!entry.dir && <span className="tree-size">{bytes(entry.size)}</span>}
    </button>
  );
}
