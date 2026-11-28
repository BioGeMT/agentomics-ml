/** Which split the diagnostics show: an iteration's validation predictions, or the test split. */
export function DiagnosticsPicker({ iterations, hasTest, selected, onSelect }: {
  iterations: number[];
  hasTest: boolean;
  selected: number | "test" | null;
  onSelect: (choice: number | "test") => void;
}) {
  return (
    <div className="diag-picker" role="radiogroup" aria-label="Show diagnostics for">
      <span>Validation of</span>
      <div className="segmented">
        {iterations.map((iteration) => (
          <button key={iteration} role="radio" aria-selected={selected === iteration} aria-checked={selected === iteration} onClick={() => onSelect(iteration)}>
            {iteration}
          </button>
        ))}
        {hasTest && <button role="radio" aria-selected={selected === "test"} aria-checked={selected === "test"} onClick={() => onSelect("test")}>Test</button>}
      </div>
    </div>
  );
}
