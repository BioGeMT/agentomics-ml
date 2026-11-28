import type { CheckResult } from "@agentomics/contract";
import { LoaderCircle } from "lucide-react";
import { DatasetSummary } from "./DatasetSummary";

/** The check inside the Docker image: in progress, what's wrong (with the original when this is a prepared copy), or a summary. */
export function DatasetCheck({ check, onUse }: { check: CheckResult | "checking"; onUse: (folder: string) => void }) {
  if (check === "checking") {
    return <p className="dataset-status checking"><LoaderCircle size={14} className="spin" aria-hidden /> Checking the dataset inside the Docker image…</p>;
  }
  if (!check.ok) {
    return (
      <div className="dataset-problem">
        <p className="dataset-status error">{check.error}</p>
        {check.suggestion && <button className="mac-secondary" onClick={() => onUse(check.suggestion!)}>Use Original</button>}
      </div>
    );
  }
  return <DatasetSummary summary={check.summary} details={check.details ?? {}} />;
}
