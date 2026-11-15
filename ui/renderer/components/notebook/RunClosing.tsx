import { Check } from "lucide-react";
import { formatDuration } from "@/model/format";

/** The line that ends a finished run's notebook. */
export function RunClosing({ done, elapsed }: { done: number; elapsed?: number }) {
  return (
    <div className="closing">
      <Check size={16} aria-hidden /> Run finished · {done} entries · {formatDuration(elapsed)}
    </div>
  );
}
