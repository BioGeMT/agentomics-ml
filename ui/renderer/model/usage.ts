// What model calls cost: per entry, per iteration and per run, from the "usage" events each response publishes.
import { addUsage, type Entry, type Usage } from "./run";

export function formatCost(usage?: Usage): string | null {
  if (!usage) return null;
  if (usage.responses === usage.unpriced) return `${formatTokens(usage.inputTokens + usage.outputTokens)} tokens`;
  const prefix = usage.unpriced > 0 ? "≥ " : "";
  if (usage.cost < 0.01) return `${prefix}<$0.01`;
  return `${prefix}$${usage.cost < 10 ? usage.cost.toFixed(2) : usage.cost.toFixed(1)}`;
}

export function formatTokens(tokens: number): string {
  if (tokens < 1000) return String(tokens);
  if (tokens < 1_000_000) return `${(tokens / 1000).toFixed(tokens < 10_000 ? 1 : 0)}k`;
  return `${(tokens / 1_000_000).toFixed(1)}M`;
}

export function usageTitle(usage?: Usage): string | undefined {
  if (!usage) return undefined;
  const parts = [
    `${usage.inputTokens.toLocaleString()} input tokens (${usage.cachedTokens.toLocaleString()} cached)`,
    `${usage.outputTokens.toLocaleString()} output tokens`,
    `${usage.responses} model ${usage.responses === 1 ? "response" : "responses"}`,
  ];
  if (usage.unpriced) parts.push(`${usage.unpriced} without a known price`);
  return parts.join(" · ");
}

export function iterationUsage(entries: Entry[], iteration: number): Usage | undefined {
  return entries.filter((e) => e.iteration === iteration).reduce<Usage | undefined>((total, e) => addUsage(total, e.usage), undefined);
}

/** ChatGPT (Codex) is a subscription: the figure is what the same tokens would cost on the API. */
export function isSubscription(provider?: string): boolean {
  return Boolean(provider && /codex/i.test(provider));
}
