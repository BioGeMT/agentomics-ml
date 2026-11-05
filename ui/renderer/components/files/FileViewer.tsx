import type { FilesResponse } from "@agentomics/contract/protocol";
import { RotateCw, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useKey } from "@/hooks/useKey";
import { bytes } from "@/model/format";
import { readWorkspace } from "@/model/workspace";
import { DownloadLink } from "./DownloadLink";
import { FileContent, type Content } from "./FileContent";
import { FileIcon } from "./FileIcon";

/** A read-only editor over the notebook: code with line numbers, tables for CSV, images inline. */
export function FileViewer({ root, rel, onClose }: { root: string; rel: string; onClose: () => void }) {
  const [content, setContent] = useState<Content | "loading">("loading");
  const load = useCallback(async () => {
    setContent("loading");
    const body: FilesResponse = await readWorkspace({ root, path: rel }, "Couldn't open this file.");
    setContent("kind" in body && body.kind === "dir" ? { error: "This is a folder." } : body);
  }, [root, rel]);
  useEffect(() => { load(); }, [load]);
  useKey("Escape", onClose);

  const name = rel.split("/").pop() ?? rel;
  return (
    <section className="file-viewer" aria-label={`File ${rel}`}>
      <header className="viewer-head">
        <FileIcon name={name} />
        <span className="viewer-path">
          {rel.split("/").slice(0, -1).map((part, i) => <span key={i} className="crumb">{part}</span>)}
          <strong>{name}</strong>
        </span>
        {content !== "loading" && "size" in content && <span className="viewer-meta">{bytes(content.size)} · read-only</span>}
        <DownloadLink root={root} rel={rel} className="viewer-button" />
        <button className="viewer-button" onClick={load} aria-label="Reload file" title="Reload"><RotateCw size={14} aria-hidden /></button>
        <button className="viewer-button" onClick={onClose} aria-label="Close file (Esc)" title="Close (Esc)"><X size={15} aria-hidden /></button>
      </header>
      <div className="viewer-body">
        <FileContent root={root} rel={rel} content={content} />
      </div>
    </section>
  );
}
