// Each finished iteration's results, read from the run's workspace (see ui/contract/snapshot.ts).
import { iterationFiles, iterationOfFolder, RUN_CONFIG, RUN_DIR, snapshotOf, type Snapshot } from "@agentomics/contract/snapshot";
import { readText, readWorkspace } from "./workspace";

const json = (text: string | null) => {
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    return null;
  }
};

export async function loadSnapshots(root: string): Promise<Snapshot[]> {
  const listing = await readWorkspace({ root, path: RUN_DIR });
  if (!("kind" in listing) || listing.kind !== "dir") return [];
  const iterations = listing.entries.map((e) => (e.dir ? iterationOfFolder(e.name) : null)).filter((i): i is number => i !== null);
  const config = json(await readText(root, RUN_CONFIG));
  const snapshots = await Promise.all(
    iterations.map(async (iteration) => {
      const files = iterationFiles(iteration);
      const [state, evaluation] = await Promise.all([readText(root, files.state), readText(root, files.evaluation)]);
      return snapshotOf(iteration, config, json(state), json(evaluation));
    }),
  );
  return snapshots.sort((a, b) => a.iteration - b.iteration);
}
