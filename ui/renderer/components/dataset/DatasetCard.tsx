import type { CheckResult } from "@agentomics/contract";
import { DatasetCheck } from "./DatasetCheck";
import { DatasetHeader } from "./DatasetHeader";
import { DatasetPrompt } from "./DatasetPrompt";

/** The chosen dataset folder and the result of checking it; before one is chosen, what to bring. */
export function DatasetCard({ folder, check, onChoose, onUse, onReveal }: {
  folder: string | null; check: CheckResult | "checking" | null; onChoose: () => void; onUse: (folder: string) => void; onReveal: (folder: string) => void;
}) {
  if (!folder) return <DatasetPrompt onChoose={onChoose} />;
  const title = (check && check !== "checking" && check.ok && check.details?.title) || folder.split("/").pop()!;
  return (
    <section className="dataset-card">
      <DatasetHeader folder={folder} title={title} onChoose={onChoose} onReveal={() => onReveal(folder)} />
      {check && <DatasetCheck check={check} onUse={onUse} />}
    </section>
  );
}
