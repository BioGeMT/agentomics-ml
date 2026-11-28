// A finished step's output as Agentomics saved it: run/iteration_N/<step>/output.json. It can differ from what
// the agent returned, because Agentomics adjusts some fields afterwards (the data split's paths point to where
// the splits were moved, run/shared/splits/split_N/).
import { stepFolder } from "./container-paths";
import type { Entry } from "./run";
import { readText } from "./workspace";

export async function loadStepOutput(root: string, entry: Entry): Promise<Record<string, unknown> | null> {
  const folder = stepFolder(entry);
  const text = folder ? await readText(root, `${folder}/output.json`) : null;
  try {
    const payload = text ? JSON.parse(text).payload : null;
    return payload && typeof payload === "object" ? payload : null;
  } catch {
    return null;
  }
}
