export type Details = "figures" | "files";

/** The segmented control between Figures and Files. */
export function DetailsTabs({ details, onDetails }: { details: Details; onDetails: (details: Details) => void }) {
  return (
    <div className="segmented" role="tablist" aria-label="Details">
      {(["figures", "files"] as const).map((name) => (
        <button key={name} role="tab" aria-selected={details === name} onClick={() => onDetails(name)}>
          {name === "figures" ? "Figures" : "Files"}
        </button>
      ))}
    </div>
  );
}
