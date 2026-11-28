export type Details = "figures" | "files" | "output";

const LABELS: Record<Details, string> = { figures: "Figures", files: "Files", output: "Output" };

/** The segmented control between Figures, Files and (in the desktop app) the runner's Output. */
export function DetailsTabs({ details, onDetails }: { details: Details; onDetails: (details: Details) => void }) {
  const tabs: Details[] = window.agentomics ? ["figures", "files", "output"] : ["figures", "files"];
  return (
    <div className="segmented" role="tablist" aria-label="Details">
      {tabs.map((name) => (
        <button key={name} role="tab" aria-selected={details === name} onClick={() => onDetails(name)}>
          {LABELS[name]}
        </button>
      ))}
    </div>
  );
}
