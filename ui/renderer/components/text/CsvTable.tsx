import { splitRow } from "@/model/text-kinds";

/** CSV or TSV as a table with row numbers, capped at limit rows. */
export function CsvTable({ text, separator = ",", limit = 500 }: { text: string; separator?: string; limit?: number }) {
  const rows = text.replace(/\n$/, "").split("\n");
  const [head, ...body] = rows.map((line) => splitRow(line, separator));
  const shown = body.slice(0, limit);
  return (
    <div className="viewer-table-wrap">
      <table className="viewer-table">
        <thead><tr><th className="row-number">#</th>{head.map((cell, i) => <th key={i}>{cell}</th>)}</tr></thead>
        <tbody>
          {shown.map((row, r) => (
            <tr key={r}><td className="row-number">{r + 1}</td>{row.map((cell, i) => <td key={i}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <p className="viewer-table-note">{body.length > shown.length ? `First ${shown.length} of ${body.length} rows` : `${body.length} rows`} · {head.length} columns</p>
    </div>
  );
}
