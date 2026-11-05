import { ChevronLeft } from "lucide-react";

/** The window title for the current screen, with a back chevron when there is somewhere to go back to. */
export function LobbyToolbar({ title, onBack }: { title: string; onBack?: () => void }) {
  return (
    <header className="lobby-toolbar">
      {onBack && (
        <button className="toolbar-back" onClick={onBack} aria-label="Back to New Run" title="Back to New Run (Esc)">
          <ChevronLeft size={18} strokeWidth={2} aria-hidden />
        </button>
      )}
      <h1>{title}</h1>
    </header>
  );
}
