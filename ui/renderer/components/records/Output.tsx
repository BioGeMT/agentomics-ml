import { useState } from "react";
import { Rendered } from "../text/Rendered";

/** A tool's output, shown as what it is: a table for CSV, highlighted JSON or code, else plain text. */
export function Output({ text, name }: { text?: string; name?: string }) {
  const [all, setAll] = useState(false);
  if (!text?.trim()) return <div className="output empty">No output</div>;
  const lines = text.replace(/\n$/, "").split("\n");
  const long = lines.length > 40;
  return (
    <div className="output">
      <Rendered text={long && !all ? lines.slice(0, 40).join("\n") : text} name={name} />
      {long && (
        <button className="fold" onClick={() => setAll(!all)}>
          {all ? "Show Less" : `Show All ${lines.length} Lines`}
        </button>
      )}
    </div>
  );
}
