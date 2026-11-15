import type { DesktopApi, Model, Provider } from "@agentomics/contract";
import { useEffect, useState } from "react";

/** The models of each connected provider, and the chosen one as "<provider>::<model id>" (the first by default). */
export function useModels(api: DesktopApi, providers: Provider[]) {
  const providerKey = providers.join(",");
  const [models, setModels] = useState<Partial<Record<Provider, Model[]>>>({});
  const [choice, setChoice] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all(providers.map(async (p) => [p, await api.listModels(p)] as const)).then((lists) => {
      if (cancelled) return;
      setModels(Object.fromEntries(lists));
      const first = lists.find(([, list]) => list.length);
      setChoice((current) => current || (first ? `${first[0]}::${first[1][0].id}` : ""));
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, providerKey]);

  return { models, loading: Object.keys(models).length === 0, choice, setChoice };
}
