// Agentomics' own AG-UI event. Everything else a run logs is a standard AG-UI event (@ag-ui/core);
// this is the one CUSTOM event, and it mirrors Usage in src/agentomics/runtime/ui_events.py.

export const USAGE_EVENT = "usage";

/** The value of the CUSTOM "usage" event: one model response's tokens and price. */
export type UsageValue = {
  model: string | null;
  provider: string | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  /** null when the model isn't in pydantic-ai's price list. */
  cost_usd: number | null;
};

const FIELDS: Record<keyof UsageValue, "string?" | "number" | "number?"> = {
  model: "string?",
  provider: "string?",
  input_tokens: "number",
  output_tokens: "number",
  cache_read_tokens: "number",
  cost_usd: "number?",
};

/** The usage event's value if it matches UsageValue exactly (no missing, extra or mistyped fields), else null. */
export function parseUsage(value: unknown): UsageValue | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record);
  if (keys.length !== Object.keys(FIELDS).length) return null;
  for (const key of keys) {
    const kind = FIELDS[key as keyof UsageValue];
    const field = record[key];
    if (!kind) return null;
    if (kind === "number" && typeof field !== "number") return null;
    if (kind === "number?" && field !== null && typeof field !== "number") return null;
    if (kind === "string?" && field !== null && typeof field !== "string") return null;
  }
  return record as UsageValue;
}
