import type { Accounts, DesktopApi } from "@agentomics/contract";
import { useState } from "react";
import { AccountRow } from "./AccountRow";
import { CodexSignIn } from "./CodexSignIn";
import { OpenRouterKeyForm } from "./OpenRouterKeyForm";
import { ReadyBadge } from "./ReadyBadge";

/** Once connected: which providers are ready, with a way to add or change them. */
export function AccountsCard({ api, accounts, onChange }: { api: DesktopApi; accounts: Accounts; onChange: () => void }) {
  const [editing, setEditing] = useState<null | "codex" | "openrouter">(null);
  const { codex, openrouter } = accounts;
  const done = () => { setEditing(null); onChange(); };
  return (
    <section className="home-card">
      <h2>Model providers</h2>
      <div className="accounts">
        <AccountRow name="ChatGPT (Codex)" meta={codex.signedIn ? "Signed in with your ChatGPT subscription" : "Not connected"}>
          {codex.signedIn ? <ReadyBadge /> : <button className="secondary" onClick={() => setEditing("codex")}>Connect</button>}
        </AccountRow>
        <AccountRow
          name="OpenRouter"
          meta={openrouter.configured ? <><code className="key">{openrouter.masked}</code>{openrouter.source === "environment" ? " · from your .env" : ""}</> : "Not connected"}
        >
          <div className="account-actions">
            {openrouter.configured && <ReadyBadge />}
            <button className="secondary" onClick={() => setEditing(editing === "openrouter" ? null : "openrouter")}>
              {openrouter.configured ? "Change Key" : "Add Key"}
            </button>
            {openrouter.source === "app" && (
              <button className="secondary" onClick={async () => { await api.removeOpenRouterKey(); onChange(); }}>Remove</button>
            )}
          </div>
        </AccountRow>
      </div>
      {editing === "codex" && <div className="account-edit"><CodexSignIn api={api} onChange={done} /></div>}
      {editing === "openrouter" && <div className="account-edit"><OpenRouterKeyForm api={api} onChange={done} /></div>}
    </section>
  );
}
