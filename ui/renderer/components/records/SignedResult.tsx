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
            <dd>{typeof v === "string" ? <LinkedText text={v} onPath={onPath} /> : <code>{JSON.stringify(v)}</code>}</dd>
          </div>
        ))}
      </dl>
      <div className="signature">
        Signed <span className="signer">Agentomics</span> {formatClock(at)}
      </div>
    </div>
  );
}
