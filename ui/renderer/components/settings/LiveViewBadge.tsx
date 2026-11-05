import { Check, LoaderCircle } from "lucide-react";

/** Whether an image can stream the live notebook: checking, yes, or no (runs still work). */
export function LiveViewBadge({ state }: { state: boolean | null | "checking" | undefined }) {
  if (state === "checking" || state === undefined) return <span className="live-badge checking"><LoaderCircle size={12} className="spin" aria-hidden /> Checking</span>;
  if (state === true) return <span className="live-badge yes"><Check size={13} aria-hidden /> Live notebook</span>;
  return (
    <span className="live-badge no" title="This image predates the live notebook. Runs work, but the app can't show their progress.">
      No live notebook
    </span>
  );
}
