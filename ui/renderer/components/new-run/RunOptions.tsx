import type { Model, Provider } from "@agentomics/contract";
import { Stepper } from "../controls/Stepper";
import { ImageWarning } from "./ImageWarning";
import { ModelPicker } from "./ModelPicker";

/** The grouped form: model, iterations and time limit, plus a warning when the image can't stream progress. */
export function RunOptions({
  providers, models, loadingModels, choice, onChoice,
  iterations, onIterations, timeoutMinutes, onTimeoutMinutes,
  image, imageLive, onOpenSettings,
}: {
  providers: Provider[]; models: Partial<Record<Provider, Model[]>>; loadingModels: boolean; choice: string; onChoice: (choice: string) => void;
  iterations: number; onIterations: (n: number) => void; timeoutMinutes: number; onTimeoutMinutes: (n: number) => void;
  image: string | null; imageLive: boolean | null | undefined; onOpenSettings: () => void;
}) {
  return (
    <section className="form-group" aria-label="Run settings">
      <ModelPicker providers={providers} models={models} loading={loadingModels} choice={choice} onChoice={onChoice} />
      <div className="form-row">
        <span id="iterations-label">Iterations</span>
        <Stepper labelledBy="iterations-label" value={iterations} min={1} max={20} onChange={onIterations} />
      </div>
      <div className="form-row">
        <span id="timeout-label">Time limit</span>
        <Stepper labelledBy="timeout-label" value={timeoutMinutes} min={5} max={1440} step={15} unit="min" onChange={onTimeoutMinutes} />
      </div>
      {imageLive === false && <ImageWarning image={image} onOpenSettings={onOpenSettings} />}
    </section>
  );
}
