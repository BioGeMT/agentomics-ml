import type { Accounts, CheckResult, DesktopApi, Provider, RunEntry } from "@agentomics/contract";
import { useEffect, useState } from "react";
import { useModels } from "@/hooks/useModels";
import { useSelectedImage } from "@/hooks/useSelectedImage";
import { availableProviders } from "@/model/accounts";
import { invokeError } from "@/model/errors";
import { DatasetCard } from "../dataset/DatasetCard";
import { ForkCard } from "./ForkCard";
import { ForkPicker } from "./ForkPicker";
import { GoalField } from "./GoalField";
import { RunOptions } from "./RunOptions";
import { StartBar } from "./StartBar";

/**
 * New Run: the dataset on one side; model, limits and goal on the other; Start Run once the dataset passes its check.
 * With a source, it forks that run instead: the run takes the dataset's place, and its goal stays.
 */
export function NewRun({ api, accounts, dropped, dockerReady, runs, source, forkAt, onOpenSettings, onStarted }: {
  api: DesktopApi; accounts: Accounts; dropped: { path: string } | null; dockerReady: boolean; runs: RunEntry[]; source?: RunEntry; forkAt?: number;
  onOpenSettings: () => void; onStarted: (entry: RunEntry) => void;
}) {
  const providers = availableProviders(accounts);
  const { models, loading, choice, setChoice } = useModels(api, providers);
  const { image, live } = useSelectedImage(api);
  const [folder, setFolder] = useState<string | null>(null);
  const [check, setCheck] = useState<CheckResult | "checking" | null>(null);
  const [iterations, setIterations] = useState(1);
  const [timeoutMinutes, setTimeoutMinutes] = useState(60);
  const [userPrompt, setUserPrompt] = useState("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const done = source?.summary?.done ?? 0;
  const [kept, setKept] = useState(forkAt && forkAt >= 1 && forkAt <= done ? forkAt : done);
  const [change, setChange] = useState("");
  // A fork starts on its source's model, once the models are in (if that model is still offered).
  const offered = Boolean(source?.provider && models[source.provider]?.some((m) => m.id === source.model));
  useEffect(() => {
    if (offered && source?.provider) setChoice(`${source.provider}::${source.model}`);
  }, [offered, source, setChoice]);

  async function loadFolder(chosen: string) {
    setFolder(chosen);
    setError(null);
    setCheck("checking");
    setCheck(await api.checkDataset(chosen, image ?? undefined));
  }

  useEffect(() => {
    if (dropped) loadFolder(dropped.path);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dropped]);

  async function chooseFolder() {
    const chosen = await api.pickFolder();
    if (chosen) loadFolder(chosen);
  }

  async function tryExample() {
    try {
      loadFolder(await api.exampleDataset());
    } catch (e) {
      setError(invokeError(e));
    }
  }

  async function start() {
    if (!(source || folder) || !choice) return;
    const [provider, model] = choice.split("::") as [Provider, string];
    setStarting(true);
    setError(null);
    try {
      onStarted(await api.startRun({
        folder: source?.folder ?? folder!, image: image ?? undefined, provider, model, iterations, timeoutMinutes, userPrompt: source ? undefined : userPrompt.trim() || undefined,
        forkFrom: source ? { workspace: source.workspace, iterations: kept, change: change.trim() || undefined } : undefined,
      }));
    } catch (e) {
      setError(invokeError(e));
      setStarting(false);
    }
  }

  const valid = source ? kept > 0 : check !== null && check !== "checking" && check.ok;
  const plural = (n: number) => `${n} iteration${n === 1 ? "" : "s"}`;
  const hint = !dockerReady
    ? "Runs start once Docker is running."
    : source
      ? `${plural(iterations)} more after iteration ${kept}, up to ${timeoutMinutes} min`
      : valid ? `${plural(iterations)}, up to ${timeoutMinutes} min` : folder ? "The dataset needs to pass the check first." : "Choose or drop a dataset folder to start.";

  return (
    <div className="new-run">
      <div className="new-run-data">
        {source
          ? <ForkCard source={source} kept={kept} onKept={setKept} />
          : <DatasetCard folder={folder} check={check} onChoose={chooseFolder} onExample={tryExample} onUse={loadFolder} onReveal={(path) => api.showFolder(path)} />}
        {!source && !folder && <ForkPicker runs={runs} />}
      </div>

      <div className="new-run-settings">
        <RunOptions
          providers={providers} models={models} loadingModels={loading} choice={choice} onChoice={setChoice}
          iterations={iterations} onIterations={setIterations} timeoutMinutes={timeoutMinutes} onTimeoutMinutes={setTimeoutMinutes}
          image={image} imageLive={live} onOpenSettings={onOpenSettings}
        />
        {source
          ? <GoalField label="What to change" value={change} onChange={setChange} placeholder={`What iteration ${kept + 1} on should try differently, e.g. "try gradient boosting instead".`} />
          : <GoalField value={userPrompt} onChange={setUserPrompt} />}
        {error && <p className="field-error">{error}</p>}
        <StartBar hint={hint} disabled={!dockerReady || !valid || !choice || starting} starting={starting} onStart={start} />
      </div>
    </div>
  );
}
