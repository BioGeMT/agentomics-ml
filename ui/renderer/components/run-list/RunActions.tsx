import type { RunEntry } from "@agentomics/contract";
import { GitFork } from "lucide-react";
import { forkLink } from "@/model/run-link";
import { SHOW_IN_FILES, TRASH } from "@/model/platform";
import { FolderSymbol } from "../symbols/FolderSymbol";
import { StopSymbol } from "../symbols/StopSymbol";
import { TrashSymbol } from "../symbols/TrashSymbol";

/** Hover actions on a run: Fork (once an iteration has finished), Stop (or Move to Trash once it has ended) and Show in Finder. */
export function RunActions({ run, onStop, onTrash, onShow }: { run: RunEntry; onStop: () => void; onTrash: () => void; onShow: () => void }) {
  return (
    <div className="run-actions">
      {(run.summary?.done ?? 0) > 0 && (
        <a href={forkLink(run.workspace)} aria-label={`Fork ${run.dataset}`} title="Fork: keep its iterations and try something different">
          <GitFork size={14} strokeWidth={2} aria-hidden />
        </a>
      )}
      {run.status === "running" ? (
        <button onClick={onStop} aria-label={`Stop ${run.dataset}`} title="Stop"><StopSymbol /></button>
      ) : (
        <button onClick={onTrash} aria-label={`Move ${run.dataset} to the ${TRASH}`} title={`Move to ${TRASH}`}><TrashSymbol /></button>
      )}
      <button onClick={onShow} aria-label={`Show ${run.dataset} files`} title={SHOW_IN_FILES}><FolderSymbol /></button>
    </div>
  );
}
