import { useEffect } from "react";
import type { Snapshot } from "@agentomics/contract/snapshot";
import type { Run } from "@/model/run";
import { loadSnapshots } from "@/model/snapshots";

/**
 * Loads the run's finished iterations from its workspace: when it opens, each time a new iteration
 * starts (the one before it is then on disk), and once more when the run ends.
 */
export function useSnapshots(root: string | null, run: Run, onSnapshots: (snapshots: Snapshot[]) => void) {
  const latest = run.entries.reduce((max, e) => Math.max(max, e.iteration), 0);
  const ended = run.status === "finished" || run.status === "failed";
  useEffect(() => {
    if (!root || run.status === "idle" || run.status === "connecting") return;
    let cancelled = false;
    loadSnapshots(root).then((snapshots) => { if (!cancelled) onSnapshots(snapshots); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, latest, ended, run.status === "idle" || run.status === "connecting"]);
}
