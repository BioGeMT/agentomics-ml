import argparse
import json
import time
from pathlib import Path

import wandb

from run_logging.logging_helpers import is_wandb_active, log_test_inference_duration
from run_logging.wandb_setup import resume_wandb_run
from runtime.conda_utils import get_best_iteration_snapshot_environment_path
from runtime.filesystem import remove_path
from runtime.inference_runner import compute_metrics, run_inference_script
from runtime.read_write_utils import load_config_from_run_dir, load_dataset_metadata
from utils.config import Config
from utils.exceptions import AgentScriptFailed
from utils.metrics import get_task_to_metrics_names
from utils.printing_utils import print_best_iteration_metrics


def _get_test_metrics_path(best_iteration_snapshot_dir: Path) -> Path:
    return best_iteration_snapshot_dir / "test_metrics.json"


def _write_test_metrics(best_iteration_snapshot_dir: Path, metrics: dict[str, float]) -> None:
    _get_test_metrics_path(best_iteration_snapshot_dir).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_test_evaluation(workspace_dir, prepared_test_sets_dir: Path):
    start = time.time()
    print("\nRunning final test evaluation...")
    config = None
    try:
        config = load_config_from_run_dir(Path(workspace_dir) / Config.RUN_DIRNAME)
        resume_wandb_run(config)
        dataset_metadata = load_dataset_metadata(config)
        best_iteration_snapshot_dir = config.best_iteration_snapshot_dir
        remove_path(_get_test_metrics_path(best_iteration_snapshot_dir))
        #TODO should use the step id by importing it
        inference_script_path = best_iteration_snapshot_dir / "model_inference" / "inference.py"
        output_path = config.best_iteration_snapshot_dir / "eval_predictions_test.csv"
        remove_path(output_path)
        prepared_test_set_dir = prepared_test_sets_dir / config.dataset
        test_input_path = prepared_test_set_dir / "test.no_label.csv"
        labeled_test_path = prepared_test_set_dir / "test.csv"
        if not test_input_path.exists() or not labeled_test_path.exists():
            raise FileNotFoundError(
                f"Expected both test.no_label.csv and test.csv in {prepared_test_set_dir} for final test evaluation."
            )
        result = run_inference_script(
            env_path=get_best_iteration_snapshot_environment_path(config),
            script_path=inference_script_path,
            input_path=test_input_path,
            output_path=output_path,
            #TODO should use the step id by importing it
            artifacts_dir=best_iteration_snapshot_dir / "model_training" / "training_artifacts",
        )
        if result.returncode != 0:
            raise AgentScriptFailed(f"Inference on test failed: {str(result)}")
        metrics = compute_metrics(
            results_file=output_path,
            labeled_input_path=labeled_test_path,
            numeric_label_col=dataset_metadata["numeric_label_col"],
            task_type=dataset_metadata["task_type"],
            evaluation_stage="test",
        )
        if metrics is not None:
            _write_test_metrics(best_iteration_snapshot_dir, metrics)
            if is_wandb_active():
                wandb.log({f"test/{metric_name}": metric_value for metric_name, metric_value in metrics.items()})
    except Exception as e:
        print("FINAL TEST EVAL FAIL", str(e))
        if is_wandb_active() and config is not None:
            wandb.log(
                {
                    f"test/{metric_name}": float("nan")
                    for metric_name in get_task_to_metrics_names()[config.task_type]
                }
            )
    log_test_inference_duration(time.time() - start)
    if config is not None:
        print_best_iteration_metrics(config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-dir', type=Path, default=Path('/workspace').resolve(), help='Path to workspace directory')
    parser.add_argument('--prepared-test-sets-dir', type=Path, required=True, help='Path to prepared test sets root directory')
    args = parser.parse_args()

    run_test_evaluation(
        args.workspace_dir,
        prepared_test_sets_dir=args.prepared_test_sets_dir.resolve(),
    )

if __name__ == "__main__":
    main()
