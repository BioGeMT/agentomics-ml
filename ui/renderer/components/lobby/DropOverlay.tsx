import { FolderOpen } from "lucide-react";

/** Shown while a folder is dragged over the window. */
export function DropOverlay() {
  return (
    <div className="drop-overlay" aria-hidden>
      <div><FolderOpen size={28} aria-hidden /><strong>Drop a dataset folder</strong><span>Agentomics checks it before the run.</span></div>
    </div>
  );
}
