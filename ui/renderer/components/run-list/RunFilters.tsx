import type { RunEntry, RunStatus } from "@agentomics/contract";
import { Search } from "lucide-react";
import { STATUS_LABEL, type RunFilter } from "@/model/run-list";

/** Search and the model and status pop-ups above the runs, as in Mail and Notes. */
export function RunFilters({ runs, filter, onChange }: { runs: RunEntry[]; filter: RunFilter; onChange: (filter: RunFilter) => void }) {
  const models = [...new Set(runs.map((r) => r.model))].sort();
  const statuses = (Object.keys(STATUS_LABEL) as RunStatus[]).filter((st) => runs.some((r) => r.status === st));
  const set = (changes: Partial<RunFilter>) => onChange({ ...filter, ...changes });
  return (
    <div className="run-filters">
      <label className="run-search">
        <Search size={13} aria-hidden />
        <input type="search" placeholder="Search" aria-label="Search runs by dataset or model" value={filter.query} onChange={(e) => set({ query: e.target.value })} />
      </label>
      <div className="run-popups">
        <select aria-label="Filter by model" value={filter.model} onChange={(e) => set({ model: e.target.value })} className={filter.model ? "active" : ""}>
          <option value="">All Models</option>
          {models.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <select aria-label="Filter by status" value={filter.status} onChange={(e) => set({ status: e.target.value })} className={filter.status ? "active" : ""}>
          <option value="">All Statuses</option>
          {statuses.map((st) => <option key={st} value={st}>{STATUS_LABEL[st]}</option>)}
        </select>
      </div>
    </div>
  );
}
