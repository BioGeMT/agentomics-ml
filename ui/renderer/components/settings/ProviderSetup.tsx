import type { DesktopApi } from "@agentomics/contract";
import { CodexSignIn } from "./CodexSignIn";
import { OpenRouterKeyForm } from "./OpenRouterKeyForm";

/** First run: connect at least one model provider before anything can run. */
export function ProviderSetup({ api, onChange }: { api: DesktopApi; onChange: () => void }) {
  return (
    <section className="home-card setup">
      <h2>Connect a model provider</h2>
      <p className="hint">Agentomics runs on a language model you already have access to. Connect one to start your first run.</p>
      <div className="provider-grid">
        <CodexSignIn api={api} onChange={onChange} />
        <OpenRouterKeyForm api={api} onChange={onChange} />
      </div>
    </section>
  );
}
