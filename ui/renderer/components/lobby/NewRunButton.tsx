import { MOD } from "@/model/platform";
import { PlusSymbol } from "../symbols/PlusSymbol";

export function NewRunButton({ onClick }: { onClick: () => void }) {
  return (
    <button className="sidebar-new" onClick={onClick} aria-label={`New Run (${MOD}N)`} title={`New Run (${MOD}N)`}>
      <PlusSymbol />
    </button>
  );
}
