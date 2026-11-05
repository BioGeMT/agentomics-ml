import type { Accounts, CheckResult, DesktopApi, Provider, RunEntry } from "@agentomics/contract";
import { useEffect, useState } from "react";
import { useModels } from "@/hooks/useModels";
import { useSelectedImage } from "@/hooks/useSelectedImage";
import { availableProviders } from "@/model/accounts";
import { invokeError } from "@/model/errors";
import { DatasetCard } from "../dataset/DatasetCard";
import { GoalField } from "./GoalField";
import { RunOptions } from "./RunOptions";
import { StartBar } from "./StartBar";

/** New Run: the dataset on one side; model, limits and goal on the other; Start Run once the dataset passes its check. */
export function NewRun({ api, accounts, dropped, dockerReady, onOpenSettings, onStarted }: {
  api: DesktopApi; accounts: Accounts; dropped: { path: string } | null; dockerReady: boolean; onOpenSettings: () => void; onStarted: (entry: RunEntry) => void;
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

  async function start() {
    if (!folder || !choice) return;
    const [provider, model] = choice.split("::") as [Provider, string];
    setStarting(true);
    setError(null);
    try {
      onStarted(await api.startRun({ folder, image: image ?? undefined, provider, model, iterations, timeoutMinutes, userPrompt: userPrompt.trim() || undefined }));
    } catch (e) {
      setError(invokeError(e));
      setStarting(false);
    }
  }

  const valid = check !== null && check !== "checking" && check.ok;
  const hint = !dockerReady ? "Runs start once Docker is running." : valid ? `${iterations} iteration${iterations === 1 ? "" : "s"}, up to ${timeoutMinutes} min` : folder ? "The dataset needs to pass the check first." : "Choose or drop a dataset folder to start.";

  return (
    <div className="new-run">
      <div className="new-run-data">
        <DatasetCard folder={folder} check={check} onChoose={chooseFolder} onUse={loadFolder} onReveal={(path) => api.showFolder(path)} />
      </div>

      <div className="new-run-settings">
        <RunOptions
          providers={providers} models={models} loadingModels={loading} choice={choice} onChoice={setChoice}
          iterations={iterations} onIterations={setIterations} timeoutMinutes={timeoutMinutes} onTimeoutMinutes={setTimeoutMinutes}
          image={image} imageLive={live} onOpenSettings={onOpenSettings}
        />
        <GoalField value={userPrompt} onChange={setUserPrompt} />
        {error && <p className="field-error">{error}</p>}
        <StartBar hint={hint} disabled={!dockerReady || !valid || !choice || starting} starting={starting} onStart={start} />
      </div>
    </div>
  );
}
