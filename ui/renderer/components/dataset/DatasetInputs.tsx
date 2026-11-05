/** The input the model trains on: how many features, in which files or folders. */
export function DatasetInputs({ summary, features }: { summary: Record<string, any>; features?: number }) {
  const structure: string[] = summary.input_structure ?? [];
  const inputs = structure.join(", ");
  if (!features && !inputs) return null;
  return (
    <p className="ds-inputs">
      <span className="ds-inputs-label">Input</span>
      {features ? <><strong>{features}</strong> features in </> : null}
      {inputs && <code>{inputs}</code>}
      {inputs && !features && structure.every((x) => x.endsWith("/")) && " folder"}
    </p>
  );
}
