// Which model providers are connected, and what to call them.
import type { Accounts, Provider } from "@agentomics/contract";

export const PROVIDER_LABEL: Record<Provider, string> = { Codex: "ChatGPT (Codex)", OpenRouter: "OpenRouter" };

/** At least one provider is ready, so runs can start. */
export function connected(accounts: Accounts): boolean {
  return accounts.codex.signedIn || accounts.openrouter.configured;
}

export function availableProviders(accounts: Accounts): Provider[] {
  return [...(accounts.codex.signedIn ? ["Codex" as const] : []), ...(accounts.openrouter.configured ? ["OpenRouter" as const] : [])];
}
