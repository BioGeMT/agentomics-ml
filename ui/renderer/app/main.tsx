// The lobby: runs, New Run and Settings.
import { createRoot } from "react-dom/client";
import { LobbyPage } from "@/components/lobby/LobbyPage";
import "@/styles/globals.css";
import "@/styles/mac-shell.css";
import "@/styles/notebook.css";
import "@/styles/lobby.css";
import "@/styles/appearance.css";

createRoot(document.getElementById("root")!).render(<LobbyPage />);
