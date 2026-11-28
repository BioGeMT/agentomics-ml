import type { RunEntry } from "@agentomics/contract";
import { CircleCheck, CircleStop, CircleX } from "lucide-react";
import { STATUS_LABEL, statusOf } from "@/model/run-list";

/** The run's state at a glance: a live dot while it runs, then a check, a cross or a stop sign. */
export function RunStatusIcon({ run }: { run: RunEntry }) {
  const status = statusOf(run);
  const label = status === "running" ? "Live" : STATUS_LABEL[status];
  const Icon = status === "finished" ? CircleCheck : status === "failed" ? CircleX : status === "interrupted" ? CircleStop : null;
  return (
    <span className={`run-status-icon ${status}`} role="img" aria-label={label} title={label}>
      {Icon ? <Icon size={13} strokeWidth={2.2} aria-hidden /> : <i />}
    </span>
  );
}
