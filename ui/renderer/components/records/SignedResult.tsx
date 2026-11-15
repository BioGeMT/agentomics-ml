import { formatClock } from "@/model/format";
import { LinkedText } from "./LinkedText";
import type { OpenPath } from "./RecordBody";

/** A step's result: the fields it returned, signed with the time it finished. */
export function SignedResult({ args, at, onPath }: { args: Record<string, any>; at?: number; onPath?: OpenPath }) {
  const entries = Object.entries(args).filter(([, v]) => v !== null && v !== "" && v !== undefined);
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
