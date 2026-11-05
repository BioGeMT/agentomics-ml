import { useState } from "react";
import { ShellOutput } from "../text/ShellOutput";

/** Output of a command or script: warnings folded, CSV lines as tables, the rest as text. */
export function ProcessOutput({ text }: { text?: string }) {
  const [all, setAll] = useState(false);
  if (!text?.trim()) return <div className="output empty">No output</div>;
  const lines = text.replace(/\n$/, "").split("\n");
  const long = lines.length > 60;
  return (
    <div className="output">
      <ShellOutput text={long && !all ? lines.slice(0, 60).join("\n") : text} />
      {long && (
        <button className="fold" onClick={() => setAll(!all)}>
          {all ? "Show Less" : `Show All ${lines.length} Lines`}
        </button>
      )}
    </div>
  );
}
