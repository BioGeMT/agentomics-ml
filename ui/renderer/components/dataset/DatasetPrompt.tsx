import { DatasetLayout } from "./DatasetLayout";

/** Before a folder is chosen: how to choose one, the example to try instead, and the layout a dataset should have. */
export function DatasetPrompt({ onChoose, onExample }: { onChoose: () => void; onExample: () => void }) {
  return (
    <section className="dataset-card dataset-empty">
      <div className="dataset-empty-head">
        <h2>Dataset</h2>
        <div className="dataset-empty-actions">
          <button className="mac-secondary" onClick={onExample}>Try an Example</button>
          <button className="mac-secondary" onClick={onChoose}>Choose Folder…</button>
        </div>
      </div>
      <p className="dataset-empty-lead">
        Choose a dataset folder, or drag one onto this window. Agentomics checks it inside the Docker image before the run.
        No dataset yet? Try the example: diagnosing breast tumours as benign or malignant from 30 measurements.
      </p>
      <DatasetLayout />
    </section>
  );
}
