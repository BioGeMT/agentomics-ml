// A finished iteration's results, read from the files Agentomics already writes for it. The lobby
// (ui/desktop) reads them from disk and the run page (ui/renderer) through /api/files; both parse them here.
//
//   run/shared/config.json                                 iterations, val_metric
//   run/iteration_<n>/runtime_info/iteration_state.json    status, started_at, ended_at
//   run/iteration_<n>/validation_evaluation/output.json    payload: metrics, is_new_best
//
// An iteration's folder appears once the iteration is over, so each one listed is finished.

export type Snapshot = {
  iteration: number; // 0-based, as Agentomics numbers them
  iterations_total: number;
  status: string;
  duration_seconds?: number;
  is_new_best: boolean;
  metrics: Record<string, number>;
  validation_metric?: string;
};

export const RUN_CONFIG = "run/shared/config.json";
export const RUN_DIR = "run";

/** The iteration number of a folder in run/, or null for anything else (shared/, current_iteration/). */
export function iterationOfFolder(name: string): number | null {
  const match = /^iteration_(\d+)$/.exec(name);
  return match ? Number(match[1]) : null;
}

export function iterationFiles(iteration: number): { state: string; evaluation: string } {
  return {
    state: `${RUN_DIR}/iteration_${iteration}/runtime_info/iteration_state.json`,
    evaluation: `${RUN_DIR}/iteration_${iteration}/validation_evaluation/output.json`,
  };
}

/** One iteration's snapshot from the parsed JSON of its files; a file that's missing is null. */
export function snapshotOf(iteration: number, config: any, state: any, evaluation: any): Snapshot {
  const payload = evaluation?.payload ?? {};
  const duration = typeof state?.started_at === "number" && typeof state?.ended_at === "number" ? state.ended_at - state.started_at : undefined;
  return {
    iteration,
    iterations_total: config?.iterations,
    status: state?.status ?? null,
    duration_seconds: duration,
    is_new_best: payload.is_new_best ?? false,
    metrics: payload.metrics ?? {},
    validation_metric: config?.val_metric ?? undefined,
  };
}
