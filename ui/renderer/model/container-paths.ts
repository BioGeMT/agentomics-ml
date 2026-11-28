// Paths the agent wrote inside its container (/workspace/…), mapped to the run's folder on disk.
import type { Entry } from "./run";
import { exists } from "./workspace";

// Step folders in a run, in step order, as they are named once an iteration is archived.
export const STEP_DIRS = ["iteration_plan", "data_exploration", "data_split", "data_representation", "model_architecture", "model_training", "model_inference", "validation_evaluation", "prediction_exploration"];

/** A finished step's archived folder in the workspace (run/iteration_N/<step>), or null for an unknown step. */
export function stepFolder(entry: Entry): string | null {
  const step = STEP_DIRS[entry.position - 1];
  return step ? `run/iteration_${entry.iteration - 1}/${step}` : null;
}

/**
 * The workspace-relative path of a /workspace/… path from an entry. Paths under current_iteration/current_step
 * only exist while a step runs; afterwards they live in run/iteration_N/<step>, so the archived place is tried too.
 */
export async function workspacePath(workspace: string, path: string, entry: Entry): Promise<string> {
  const rel = path.replace(/^\/workspace\/?/, "");
  const iteration = `run/iteration_${entry.iteration - 1}`;
  const step = STEP_DIRS[entry.position - 1];
  const archived = rel
    .replace(/^run\/current_iteration\/current_step(?=\/|$)/, step ? `${iteration}/${step}` : iteration)
    .replace(/^run\/current_iteration(?=\/|$)/, iteration);
  let chosen = rel;
  for (const candidate of archived === rel ? [rel] : [rel, archived]) {
    if (await exists(workspace, candidate)) { chosen = candidate; break; }
    chosen = archived;
  }
  return chosen;
}
