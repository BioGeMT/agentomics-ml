import { File, FileCode, FileText, Image as ImageIcon, Sheet } from "lucide-react";
import { extension, IMAGE_EXTENSIONS } from "@/model/text-kinds";

/** A file's icon by its kind: code, a table, an image, a document, else a plain file. */
export function FileIcon({ name }: { name: string }) {
  const ext = extension(name);
  if (["py", "sh", "json", "yaml", "yml"].includes(ext)) return <FileCode size={14} aria-hidden />;
  if (["csv", "tsv"].includes(ext)) return <Sheet size={14} aria-hidden />;
  if (IMAGE_EXTENSIONS.includes(ext)) return <ImageIcon size={14} aria-hidden />;
  if (ext === "pdf") return <FileText size={14} aria-hidden />;
  if (["md", "txt", "log"].includes(ext)) return <FileText size={14} aria-hidden />;
  return <File size={14} aria-hidden />;
}
