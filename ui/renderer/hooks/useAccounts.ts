import type { Accounts, DesktopApi } from "@agentomics/contract";
import { useCallback, useEffect, useState } from "react";

/** The model provider accounts; null until loaded. */
export function useAccounts(api: DesktopApi): [Accounts | null, () => Promise<void>] {
  const [accounts, setAccounts] = useState<Accounts | null>(null);
  const refresh = useCallback(async () => setAccounts(await api.getAccounts()), [api]);
  useEffect(() => { refresh(); }, [refresh]);
  return [accounts, refresh];
}
