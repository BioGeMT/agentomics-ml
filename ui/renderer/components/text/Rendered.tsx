import { extension, languageFor, looksLikeCsv, prettyJson } from "@/model/text-kinds";
import { CodeBlock } from "./CodeBlock";
import { CsvTable } from "./CsvTable";

/**
 * Picks the best view for a block of text. `name` is a file name or path the text came from, when known
 * (a written file, or the file a command printed), and decides first; otherwise the content does.
 */
export function Rendered({ text, name, numbers = false }: { text: string; name?: string; numbers?: boolean }) {
  const ext = name ? extension(name) : "";
  if ((ext === "csv" || ext === "tsv") && (ext === "tsv" || looksLikeCsv(text))) return <CsvTable text={text} separator={ext === "tsv" ? "\t" : ","} limit={200} />;
  const language = name ? languageFor(name) : undefined;
  if (language === "json" || !language) {
    const pretty = prettyJson(text);
    if (pretty) return <CodeBlock text={pretty} language="json" numbers={numbers} />;
  }
  if (language) return <CodeBlock text={text} language={language} numbers={numbers} />;
  return <CodeBlock text={text} numbers={numbers} />;
}
