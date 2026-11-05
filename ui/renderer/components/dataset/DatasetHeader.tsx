import { segments, SHOW_IN_FILES } from "@/model/platform";
import { FinderIcon } from "../symbols/FinderIcon";

/** The dataset's name, where it is (as Finder would say it; the full path is in the tooltip), and Change…. */
export function DatasetHeader({ folder, title, onChoose, onReveal }: { folder: string; title: string; onChoose: () => void; onReveal: () => void }) {
  return (
    <div className="dataset-head">
      <span className="dataset-icon"><FinderIcon folder /></span>
      <div>
        <strong>{title}</strong>
        <button className="dataset-where" onClick={onReveal} title={`${folder}\n${SHOW_IN_FILES}`}>
          in {segments(folder).slice(-2, -1)[0]} <span aria-hidden>›</span> {segments(folder).pop()}
        </button>
      </div>
      <button className="mac-secondary" onClick={onChoose}>Change…</button>
    </div>
  );
}
