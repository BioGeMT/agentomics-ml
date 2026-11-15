import type { ImageOption, PullProgress } from "@agentomics/contract";
import { Circle, CircleDot, Download, LoaderCircle } from "lucide-react";
import { LiveViewBadge } from "./LiveViewBadge";

export type PullState = PullProgress | { error: string };

/** One image: a radio to choose it, then whether it streams the live notebook, or Download with progress. */
export function ImageRow({ image, meta, selected, live, pull, onChoose, onDownload }: {
  image: ImageOption;
  meta: string;
  selected: boolean;
  live: boolean | null | "checking" | undefined;
  pull?: PullState;
  onChoose: () => void;
  onDownload: () => void;
}) {
  const available = image.downloaded !== false;
  return (
    <li className={`image-row ${selected ? "selected" : ""}`}>
      <button className="image-choose" onClick={onChoose} disabled={!available || selected} aria-pressed={selected}>
        {selected ? <CircleDot size={16} aria-hidden /> : <Circle size={16} aria-hidden />}
        <span className="image-name">{image.tag ?? image.name}</span>
        <span className="image-meta">{meta}</span>
      </button>
      <div className="image-state">
        {available && <LiveViewBadge state={live} />}
        {!available && !pull && (
          <button className="mac-secondary" onClick={onDownload}><Download size={13} aria-hidden /> Download</button>
        )}
        {pull && "error" in pull && <span className="image-error">{pull.error}</span>}
        {pull && !("error" in pull) && (
          <span className="image-progress">
            <LoaderCircle size={13} className="spin" aria-hidden />
            {pull.total ? `Downloading · ${pull.done} of ${pull.total} layers` : "Starting download…"}
          </span>
        )}
      </div>
    </li>
  );
}
