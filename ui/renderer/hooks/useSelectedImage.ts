import type { DesktopApi } from "@agentomics/contract";
import { useEffect, useState } from "react";

/** The Docker image new runs use, and whether it streams the live notebook (undefined while checking). */
export function useSelectedImage(api: DesktopApi): { image: string | null; live: boolean | null | undefined } {
  const [image, setImage] = useState<string | null>(null);
  const [live, setLive] = useState<boolean | null | undefined>(undefined);
  useEffect(() => {
    api.listImages().then((result) => {
      setImage(result.selected);
      const known = [...result.official, ...result.local].find((i) => i.name === result.selected);
      if (known?.live !== undefined) setLive(known.live);
      else api.checkLiveView(result.selected).then(setLive);
    });
  }, [api]);
  return { image, live };
}
