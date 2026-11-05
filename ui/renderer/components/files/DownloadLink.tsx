import { Download } from "lucide-react";
import { downloadUrl } from "@/model/workspace";

/** Saves a workspace file, or a folder as a zip: the desktop app asks where with a Save dialog, a browser downloads it. */
export function DownloadLink({ root, rel, folder = false, className }: { root: string; rel: string; folder?: boolean; className: string }) {
  const name = rel.split("/").pop() ?? rel;
  const saved = folder ? `${name}.zip` : name;
  return (
    <a className={className} href={downloadUrl(root, rel)} download={saved} aria-label={`Download ${saved}`} title={folder ? "Download as Zip…" : "Download…"}>
      <Download size={14} aria-hidden />
    </a>
  );
}
