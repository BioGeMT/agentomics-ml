import { ArrowDown } from "lucide-react";
import type { Entry } from "@/model/run";

/** A frosted capsule back to the live step, shown when the reader has scrolled away from it. */
export function JumpToLive({ entry, onJump }: { entry: Entry; onJump: () => void }) {
  return (
    <button className="jump-live" onClick={onJump} aria-label={`Jump to the live step, ${entry.key} ${entry.title}`}>
      <span className="jump-dot" aria-hidden />
      <span className="jump-label">Live</span>
      <span className="jump-step">{entry.key} {entry.title}</span>
      <ArrowDown size={14} strokeWidth={2.2} aria-hidden className="jump-arrow" />
    </button>
  );
}
