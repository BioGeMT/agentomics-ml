import type { DesktopApi, RunEntry } from "@agentomics/contract";
import { useCallback, useEffect, useState } from "react";

/** The run history, refreshed every few seconds so running runs advance. */
export function useRuns(api: DesktopApi): [RunEntry[], () => Promise<void>] {
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const refresh = useCallback(async () => setRuns(await api.listRuns()), [api]);
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [refresh]);
  return [runs, refresh];
}
