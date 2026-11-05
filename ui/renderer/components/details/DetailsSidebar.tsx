import type { EvaluationState } from "@/hooks/useEvaluation";
import type { Run } from "@/model/run";
import { FileTree } from "../files/FileTree";
import { DetailsTabs, type Details } from "./DetailsTabs";
import { FiguresPanel } from "./FiguresPanel";

/** The right sidebar: the run's figures, or its workspace files when the run's folder is known. */
export function DetailsSidebar({ run, workspace, details, onDetails, evaluation, provider, openFile, now, hidden, inert, onOpenFile, onGoTo }: {
  run: Run;
  workspace: string | null;
  details: Details;
  onDetails: (details: Details) => void;
  evaluation: EvaluationState;
  provider?: string;
  openFile: string | null;
  now: number;
  hidden: boolean;
  inert: boolean;
  onOpenFile: (rel: string) => void;
  onGoTo: (key: string) => void;
}) {
  return (
    <aside className="figures" id="panel-figures" aria-label="Figures" hidden={hidden} inert={inert}>
      <div className="side-head">
        {workspace ? <DetailsTabs details={details} onDetails={onDetails} /> : <h2>Figures</h2>}
      </div>
      {details === "files" && workspace ? (
        <FileTree root={workspace} live={run.status === "running"} selected={openFile} onOpen={onOpenFile} />
      ) : (
        <FiguresPanel run={run} evaluation={evaluation} provider={provider} now={now} onGoTo={onGoTo} />
      )}
    </aside>
  );
}
