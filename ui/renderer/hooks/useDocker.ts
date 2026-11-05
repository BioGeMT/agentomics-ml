import type { DesktopApi, DockerStatus } from "@agentomics/contract";
import { useCallback, useEffect, useState } from "react";

/** Docker's status: null until known, then checked every few seconds until the engine answers. */
export function useDocker(api: Pick<DesktopApi, "dockerStatus">): [DockerStatus | null, () => Promise<void>] {
  const [status, setStatus] = useState<DockerStatus | null>(null);
  const check = useCallback(async () => setStatus(await api.dockerStatus()), [api]);
  useEffect(() => { check(); }, [check]);
  // The next check starts 4 s after the last one answered: docker info can take seconds while Docker starts.
  useEffect(() => {
    if (!status || status.running) return;
    const timer = setTimeout(check, 4000);
    return () => clearTimeout(timer);
  }, [status, check]);
  return [status, check];
}
