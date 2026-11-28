import { useState } from "react";
import { formatClock } from "@/model/format";
import { LinkedText } from "./LinkedText";
import type { OpenPath } from "./RecordBody";

/**
 * A step's result: the fields it returned, signed with the time it finished. saved: the step's output as
 * Agentomics saved it; where it has a field of the same kind, its value wins (e.g. the data split's paths,
 * which point to where the splits were moved).
 */
export function SignedResult({ args, saved, at, onPath }: { args: Record<string, any>; saved?: Record<string, unknown> | null; at?: number; onPath?: OpenPath }) {
  const entries = Object.entries(args)
    .map(([key, v]): [string, any] => [key, saved && typeof saved[key] === typeof v ? saved[key] : v])
    .filter(([, v]) => v !== null && v !== "" && v !== undefined);
  return (
    <div className="result">
      <dl>
        {entries.map(([key, v]) => (
          <div key={key} className="result-row">
            <dt>{key.replaceAll("_", " ")}</dt>
            <dd>{typeof v === "string" ? <Field text={v} onPath={onPath} /> : <code>{JSON.stringify(v)}</code>}</dd>
          </div>
        ))}
      </dl>
      <div className="signature">
        Signed <span className="signer">Agentomics</span> {formatClock(at)}
      </div>
    </div>
  );
}

// Past this many characters a field opens folded to its first lines, so a step's result reads as a summary.
const LONG_FIELD = 480;

/** A text field of the result; a long one folded, with the reader's choice to read it all. */
function Field({ text, onPath }: { text: string; onPath?: OpenPath }) {
  const [open, setOpen] = useState(false);
  if (text.length <= LONG_FIELD) return <LinkedText text={text} onPath={onPath} />;
  return (
    <>
      <div className={open ? "field-text" : "field-text folded"}><LinkedText text={text} onPath={onPath} /></div>
      <button className="field-fold" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "Show less" : "Show more"}</button>
    </>
  );
}
