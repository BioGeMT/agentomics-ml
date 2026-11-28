import { useMemo } from "react";
import { highlight } from "@/model/highlight";

/** Highlighted source with a line-number gutter. */
export function CodeBlock({ text, language, numbers = true }: { text: string; language?: string; numbers?: boolean }) {
  const html = useMemo(() => (!language || text.length > 400_000 ? null : highlight(text, language)), [text, language]);
  const lines = text.replace(/\n$/, "").split("\n").length;
  return (
    <div className="viewer-code">
      {numbers && <pre className="gutter" aria-hidden>{Array.from({ length: lines }, (_, i) => i + 1).join("\n")}</pre>}
      {html !== null ? <pre className="source hljs" dangerouslySetInnerHTML={{ __html: html }} /> : <pre className="source">{text}</pre>}
    </div>
  );
}
