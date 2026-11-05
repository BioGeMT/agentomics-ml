import type { DesktopApi } from "@agentomics/contract";
import { useCallback, useEffect, useState } from "react";
import { useKey } from "./useKey";

export type LobbyView = "new" | "settings";

/** Which screen the lobby shows. The menu bar (⌘N, ⌘,) switches it; Escape leaves Settings, like a sheet. */
export function useLobbyView(api: DesktopApi): [LobbyView, (view: LobbyView) => void] {
  const [view, setView] = useState<LobbyView>(() => (typeof window !== "undefined" && window.location.hash === "#settings" ? "settings" : "new"));
  useEffect(() => api.onMenu((action) => {
    if (action === "new-run") setView("new");
    if (action === "settings") setView("settings");
  }), [api]);
  useKey("Escape", useCallback(() => setView("new"), []), view === "settings");
  return [view, setView];
}
