import type { DesktopApi } from "@agentomics/contract";
import { LoaderCircle, LogIn } from "lucide-react";
import { useState } from "react";

/** ChatGPT (Codex): sign in with the user's subscription, in their browser. */
export function CodexSignIn({ api, onChange }: { api: Pick<DesktopApi, "codexSignIn">; onChange: () => void }) {
  const [state, setState] = useState<"idle" | "waiting" | { error: string }>("idle");
  async function signIn() {
    setState("waiting");
    const result = await api.codexSignIn();
    if (result.ok) onChange();
    else setState({ error: result.error ?? "Sign-in didn't finish." });
  }
  return (
    <div className="provider">
      <h3>ChatGPT (Codex)</h3>
      <p>Use your ChatGPT subscription. Sign-in happens in your browser; nothing to copy or paste.</p>
      <button className="primary" onClick={signIn} disabled={state === "waiting"}>
        {state === "waiting" ? <LoaderCircle size={15} className="spin" aria-hidden /> : <LogIn size={15} aria-hidden />}
        {state === "waiting" ? "Finish signing in in your browser…" : "Sign In with ChatGPT"}
      </button>
      {typeof state === "object" && <p className="field-error">{state.error}</p>}
    </div>
  );
}
