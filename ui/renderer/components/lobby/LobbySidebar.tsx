import type { DesktopApi, RunEntry } from "@agentomics/contract";
import type { LobbyView } from "@/hooks/useLobbyView";
import { RunList } from "../run-list/RunList";
import { NewRunButton } from "./NewRunButton";
import { SettingsButton } from "./SettingsButton";

export function LobbySidebar({ api, runs, onRunsChange, view, onView, showSettings }: {
  api: DesktopApi;
  runs: RunEntry[];
  onRunsChange: () => void;
  view: LobbyView;
  onView: (view: LobbyView) => void;
  showSettings: boolean;
}) {
  return (
    <aside className="lobby-sidebar">
      <div className="sidebar-top">
        <NewRunButton onClick={() => onView("new")} />
      </div>
      <h2 className="sidebar-section">Runs</h2>
      <RunList api={api} runs={runs} onChange={onRunsChange} />
      {showSettings && (
        <footer className="sidebar-footer">
          <SettingsButton selected={view === "settings"} onClick={() => onView(view === "settings" ? "new" : "settings")} />
        </footer>
      )}
    </aside>
  );
}
