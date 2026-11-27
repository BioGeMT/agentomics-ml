import { useEffect, useState } from "react";
import type { Entry } from "@/model/run";
import { loadStepOutput } from "@/model/step-output";

/** A finished step's saved output (model/step-output.ts), loaded once the step is done; null until then. */
export function useStepOutput(root: string | null, entry: Entry): Record<string, unknown> | null {
  const [output, setOutput] = useState<Record<string, unknown> | null>(null);
  const done = entry.status === "done";
  useEffect(() => {
    if (!root || !done) return;
    let cancelled = false;
    loadStepOutput(root, entry).then((saved) => { if (!cancelled) setOutput(saved); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, done, entry.key]);
  return output;
}
