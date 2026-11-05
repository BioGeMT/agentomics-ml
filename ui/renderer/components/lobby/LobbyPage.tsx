import { useEffect, useState } from "react";
import { Lobby } from "./Lobby";
import { OpenDesktopApp } from "./OpenDesktopApp";

/** The lobby needs the desktop app's bridge; in a plain browser it says how to get it. */
export function LobbyPage() {
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  if (!ready) return null;
  const api = window.agentomics;
  return api ? <Lobby api={api} /> : <OpenDesktopApp />;
}
