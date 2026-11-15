import { MOD } from "@/model/platform";
import { GearSymbol } from "../symbols/GearSymbol";

/** Settings at the foot of the sidebar; selected while Settings is open. */
export function SettingsButton({ selected, onClick }: { selected: boolean; onClick: () => void }) {
  return (
    <button className={`sidebar-settings ${selected ? "selected" : ""}`} onClick={onClick} aria-pressed={selected} title={`Settings (${MOD},)`}>
      <GearSymbol /> Settings
    </button>
  );
}
