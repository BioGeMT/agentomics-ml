import { DatasetLayout } from "./DatasetLayout";

/** Before a folder is chosen: how to choose one, and the layout it should have. */
export function DatasetPrompt({ onChoose }: { onChoose: () => void }) {
  return (
    <section className="dataset-card dataset-empty">
      <div className="dataset-empty-head">
        <h2>Dataset</h2>
        <button className="mac-secondary" onClick={onChoose}>Choose Folder…</button>
      </div>
      <p className="dataset-empty-lead">Choose a dataset folder, or drag one onto this window. Agentomics checks it inside the Docker image before the run.</p>
      <DatasetLayout />
    </section>
  );
}
