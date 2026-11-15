import { Check, Circle, LoaderCircle, Minus, X } from "lucide-react";
import type { Entry } from "@/model/run";

/** An entry's state as a small mark: done, in progress, failed, skipped or upcoming. */
export function StatusMark({ status }: { status: Entry["status"] }) {
  if (status === "done") return <Check size={13} className="mark done" aria-label="done" />;
  if (status === "running") return <LoaderCircle size={13} className="mark running spin" aria-label="in progress" />;
  if (status === "failed") return <X size={13} className="mark failed" aria-label="failed" />;
  if (status === "skipped") return <Minus size={13} className="mark skipped" aria-label="skipped" />;
  return <Circle size={9} className="mark upcoming" aria-label="upcoming" />;
}
