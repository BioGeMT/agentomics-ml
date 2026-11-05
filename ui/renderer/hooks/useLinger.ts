import { useEffect, useState } from "react";

/** True while `open`, and for the length of the slide animation after it turns false. */
export function useLinger(open: boolean, ms = 340): boolean {
  const [lingering, setLingering] = useState(open);
  useEffect(() => {
    if (open) {
      setLingering(true);
      return;
    }
    const timer = setTimeout(() => setLingering(false), ms);
    return () => clearTimeout(timer);
  }, [open, ms]);
  return lingering;
}
