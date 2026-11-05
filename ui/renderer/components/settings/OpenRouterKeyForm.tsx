import type { DesktopApi } from "@agentomics/contract";
import { KeyRound, LoaderCircle } from "lucide-react";
import { useState } from "react";

/** OpenRouter: paste an API key; the app checks it and keeps it in the system keychain. */
export function OpenRouterKeyForm({ api, onChange }: { api: Pick<DesktopApi, "saveOpenRouterKey">; onChange: () => void }) {
  const [key, setKey] = useState("");
  const [state, setState] = useState<"idle" | "checking" | { error: string }>("idle");
  async function save(event: React.FormEvent) {
    event.preventDefault();
    setState("checking");
    const result = await api.saveOpenRouterKey(key);
    if (result.ok) {
      setKey("");
      onChange();
    } else setState({ error: result.error ?? "The key couldn't be saved." });
  }
  return (
    <form className="provider" onSubmit={save}>
      <h3>OpenRouter</h3>
      <p>One API key for models from many providers. Create one at <span className="select">openrouter.ai/keys</span>.</p>
      <label className="visually-hidden" htmlFor="openrouter-key">OpenRouter API key</label>
      <input id="openrouter-key" type="password" autoComplete="off" spellCheck={false} placeholder="sk-or-v1-…" value={key} onChange={(e) => setKey(e.target.value)} />
      <button className="primary" type="submit" disabled={!key.trim() || state === "checking"}>
        {state === "checking" ? <LoaderCircle size={15} className="spin" aria-hidden /> : <KeyRound size={15} aria-hidden />}
        {state === "checking" ? "Checking the key…" : "Save Key"}
      </button>
      {typeof state === "object" && <p className="field-error">{state.error}</p>}
      <p className="fine">Stored encrypted in your system keychain and only passed to runs you start.</p>
    </form>
  );
}
