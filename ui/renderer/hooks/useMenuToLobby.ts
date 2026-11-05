import { useEffect } from "react";

/** File › New Run and Settings… from the menu bar lead back to the lobby. */
export function useMenuToLobby(): void {
  useEffect(() => window.agentomics?.onMenu((action) => {
    if (action === "new-run") window.location.href = "/app/";
    if (action === "settings") window.location.href = "/app/#settings";
  }), []);
}
