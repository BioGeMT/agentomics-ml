// A run as a lab notebook, opened with a run link (model/run-link.ts).
import { createRoot } from "react-dom/client";
import { RunPage } from "@/components/run/RunPage";
// The same cascade as before: base and window chrome, then the page's sheets, appearance.css last.
import "@/styles/globals.css";
import "@/styles/mac-shell.css";
import "@/styles/files.css";
import "@/styles/diagnostics.css";
import "@/styles/notebook.css";
import "@/styles/appearance.css";

createRoot(document.getElementById("root")!).render(<RunPage />);
