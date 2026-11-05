import type { Model, Provider } from "@agentomics/contract";
import { PROVIDER_LABEL } from "@/model/accounts";

/** The model pop-up, grouped by provider. Values are "<provider>::<model id>". */
export function ModelPicker({ providers, models, loading, choice, onChoice }: {
  providers: Provider[]; models: Partial<Record<Provider, Model[]>>; loading: boolean; choice: string; onChoice: (choice: string) => void;
}) {
  return (
    <label className="form-row">
      <span>Model</span>
      <select value={choice} onChange={(e) => onChoice(e.target.value)} disabled={loading}>
        {loading && <option>Loading models…</option>}
        {providers.map((p) => (
          <optgroup key={p} label={PROVIDER_LABEL[p]}>
            {(models[p] ?? []).map((m) => <option key={m.id} value={`${p}::${m.id}`}>{m.label}</option>)}
          </optgroup>
        ))}
      </select>
    </label>
  );
}
