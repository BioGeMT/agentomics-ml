import type { FilesResponse } from "@agentomics/contract/protocol";
import { LoaderCircle } from "lucide-react";
import { bytes } from "@/model/format";
import { SHOW_IN_FILES } from "@/model/platform";
import { extension, languageFor } from "@/model/text-kinds";
import { rawUrl } from "@/model/workspace";
import { CodeBlock } from "../text/CodeBlock";
import { CsvTable } from "../text/CsvTable";

export type Content = Exclude<FilesResponse, { kind: "dir" }>;

/** A file shown as what it is: an image, a PDF, a table, code, or why it can't be shown. */
export function FileContent({ root, rel, content }: { root: string; rel: string; content: Content | "loading" }) {
  const name = rel.split("/").pop() ?? rel;
  const ext = extension(name);
  const showInFinder = typeof window !== "undefined" ? window.agentomics?.showFolder : undefined;
  if (content === "loading") return <p className="viewer-note"><LoaderCircle size={14} className="spin" aria-hidden /> Opening…</p>;
  if ("error" in content) return <p className="viewer-note error">{content.error}</p>;
  switch (content.kind) {
    case "image":
      return <div className="viewer-image"><img src={rawUrl(root, rel)} alt={name} /></div>;
    case "pdf":
      return <iframe className="viewer-pdf" src={rawUrl(root, rel)} title={name} />;
    case "binary":
    case "large":
      return (
        <div className="viewer-note">
          <p>{content.kind === "large" ? `This file is ${bytes(content.size)}, too large to show here.` : "This is a binary file and can't be shown as text."}</p>
          {showInFinder && <button className="mac-secondary" onClick={() => showInFinder(`${root}/${rel.split("/").slice(0, -1).join("/")}`)}>{SHOW_IN_FILES}</button>}
        </div>
      );
    case "text":
      return ["csv", "tsv"].includes(ext) ? <CsvTable text={content.text} separator={ext === "tsv" ? "\t" : ","} /> : <CodeBlock text={content.text} language={languageFor(name)} />;
  }
}
