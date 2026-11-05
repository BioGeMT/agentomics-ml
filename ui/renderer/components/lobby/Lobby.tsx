import type { DesktopApi } from "@agentomics/contract";
import { useCallback, useState } from "react";
import { useAccounts } from "@/hooks/useAccounts";
import { useDocker } from "@/hooks/useDocker";
import { useFolderDrop } from "@/hooks/useFolderDrop";
import { useLobbyView } from "@/hooks/useLobbyView";
import { useRuns } from "@/hooks/useRuns";
import { connected } from "@/model/accounts";
import { runLink } from "@/model/run-link";
import { DockerSetup } from "../new-run/DockerSetup";
import { NewRun } from "../new-run/NewRun";
import { ProviderSetup } from "../settings/ProviderSetup";
import { Settings } from "../settings/Settings";
import { DropOverlay } from "./DropOverlay";
import { LobbySidebar } from "./LobbySidebar";
import { LobbyToolbar } from "./LobbyToolbar";

/** The app's home: runs on the left; New Run, Settings or first-run setup on the right. */
export function Lobby({ api }: { api: DesktopApi }) {
  const [runs, refreshRuns] = useRuns(api);
  const [accounts, refreshAccounts] = useAccounts(api);
  const [docker] = useDocker(api);
  const [view, setView] = useLobbyView(api);
  const [dropped, setDropped] = useState<{ path: string } | null>(null);
  const onDrop = useCallback((path: string) => {
    setView("new");
    setDropped({ path });
  }, [setView]);
  const dragging = useFolderDrop(api, onDrop);
  const ready = accounts !== null && connected(accounts);

  return (
    <div className="lobby">
      <LobbySidebar api={api} runs={runs} onRunsChange={refreshRuns} view={view} onView={setView} showSettings={ready} />

      <main className="lobby-main">
        <LobbyToolbar
          title={accounts && !ready ? "Get Started" : view === "settings" ? "Settings" : "New Run"}
          onBack={ready && view === "settings" ? () => setView("new") : undefined}
        />
        <div className="lobby-scroll">
          <div className="lobby-content home">
            {accounts && !ready && <ProviderSetup api={api} onChange={refreshAccounts} />}
            {accounts && ready && view === "new" && docker && !docker.running && <DockerSetup status={docker} api={api} />}
            {accounts && ready && view === "new" && (
              <NewRun
                api={api} accounts={accounts} dropped={dropped} dockerReady={Boolean(docker?.running)}
                onOpenSettings={() => setView("settings")} onStarted={(entry) => (window.location.href = runLink(entry))}
              />
            )}
            {accounts && ready && view === "settings" && <Settings api={api} accounts={accounts} onAccountsChange={refreshAccounts} />}
          </div>
        </div>
      </main>

      {dragging && <DropOverlay />}
    </div>
  );
}
