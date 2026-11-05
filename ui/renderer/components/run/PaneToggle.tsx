import type { Side } from "@/hooks/usePanes";
import { MOD } from "@/model/platform";
import { SidebarSymbol } from "../symbols/SidebarSymbol";

const LABEL: Record<Side, { aria: string; title: string; symbol: "left" | "right" }> = {
  contents: { aria: `Show or hide contents (${MOD}[)`, title: `Contents (${MOD}[)`, symbol: "left" },
  figures: { aria: `Show or hide details (${MOD}])`, title: `Details (${MOD}])`, symbol: "right" },
};

/** The toolbar button that shows or hides a sidebar. */
export function PaneToggle({ side, open, onToggle }: { side: Side; open: boolean; onToggle: () => void }) {
  const label = LABEL[side];
  return (
    <button className={`toolbar-toggle toggle-${side}`} onClick={onToggle} aria-pressed={open} aria-label={label.aria} title={label.title}>
      <SidebarSymbol side={label.symbol} />
    </button>
  );
}
