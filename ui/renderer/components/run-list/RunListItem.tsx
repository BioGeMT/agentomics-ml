import type { RunEntry } from "@agentomics/contract";
import { runDetail } from "@/model/run-list";
import { runLink } from "@/model/run-link";
import { IterationDots } from "./IterationDots";
import { RunActions } from "./RunActions";
import { RunScore } from "./RunScore";

export function RunListItem({ run, onMenu, onStop, onTrash, onShow }: {
  run: RunEntry;
  onMenu: (event: React.MouseEvent) => void;
  onStop: () => void;
  onTrash: () => void;
  onShow: () => void;
}) {
  return (
    <li onContextMenu={onMenu}>
      <a className="run-item" href={runLink(run)} title={`${run.model} · ${run.folder}`}>
        <span className={`run-dot ${run.status}`} aria-label={run.status === "running" ? "live" : run.status} />
        <span className="run-name">{run.dataset}</span>
        <RunScore run={run} />
        <span className="run-sub"><IterationDots run={run} />{runDetail(run)}</span>
        <span className="run-when">{new Date(run.startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · {run.model}</span>
      </a>
      <RunActions run={run} onStop={onStop} onTrash={onTrash} onShow={onShow} />
    </li>
  );
}
