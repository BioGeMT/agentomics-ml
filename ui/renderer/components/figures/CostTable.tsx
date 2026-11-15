import { addUsage, type Entry, type Usage } from "@/model/run";
import { formatCost, formatTokens, isSubscription, iterationUsage, usageTitle } from "@/model/usage";

/** Table 2: tokens and cost per iteration, with the run's total. */
export function CostTable({ entries, provider }: { entries: Entry[]; provider?: string }) {
  const withUsage = entries.filter((e) => e.usage);
  if (withUsage.length === 0) return null;
  const total = withUsage.reduce<Usage | undefined>((sum, e) => addUsage(sum, e.usage), undefined)!;
  const iterations = [...new Set(withUsage.map((e) => e.iteration))].sort((a, b) => a - b);
  const subscription = isSubscription(provider ?? total.provider);

  return (
    <figure className="figure cost">
      <figcaption>
        <span className="fig-label">Table 2</span> Cost
        <span className="cost-total" title={usageTitle(total)}>{formatCost(total)}</span>
      </figcaption>
      {subscription && <p className="figure-note">API-equivalent: your ChatGPT plan isn&apos;t billed per token.</p>}
      <table className="metrics-table cost-table">
        <thead><tr><th>Iteration</th><th>Tokens</th><th>Cost</th></tr></thead>
        <tbody>
          {iterations.map((it) => {
            const u = iterationUsage(entries, it)!;
            return (
              <tr key={it} title={usageTitle(u)}>
                <th>{it}</th>
                <td>{formatTokens(u.inputTokens + u.outputTokens)}</td>
                <td>{formatCost(u)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </figure>
  );
}
