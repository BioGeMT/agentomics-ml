import argparse
import json
import tempfile
import time
from pathlib import Path

import wandb

from datasets.data_contract import INPUT_DIR_NAME, LABELS_FILE_NAME, TEST_SPLIT, NUMERIC_LABEL_COLUMN_NAME
from datasets.dataset_preparation import prepare_test_dataset
from run_logging.logging_helpers import is_wandb_active, log_test_inference_duration
from run_logging.wandb_setup import resume_wandb_run
from runtime.conda_utils import create_environment_from_descriptor
from runtime.filesystem import remove_path
from runtime.inference_runner import compute_metrics, run_inference_script
from runtime.read_write_utils import load_config_from_run_dir
from utils.config import Config
from utils.exceptions import AgentScriptFailed
from utils.metrics import get_task_to_metrics_names
from utils.printing_utils import print_best_iteration_metrics


def _get_test_metrics_path(best_iteration_snapshot_dir: Path) -> Path:
    return best_iteration_snapshot_dir / "test_metrics.json"


def _write_test_metrics(best_iteration_snapshot_dir: Path, metrics: dict[str, float]) -> None:
    _get_test_metrics_path(best_iteration_snapshot_dir).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_test_evaluation(workspace_dir, test_datasets_dir: Path):
    start = time.time()
    print("\nRunning final test evaluation...")
    config = load_config_from_run_dir(Path(workspace_dir) / Config.RUN_DIRNAME)
    test_eval_env_dir = config.shared_dir / "_test_eval_env"
    raw_test_dataset_dir = test_datasets_dir / config.dataset
    raw_test_split_path = raw_test_dataset_dir / TEST_SPLIT
    raw_test_input_path = raw_test_split_path / INPUT_DIR_NAME
    raw_labels_path = raw_test_split_path / LABELS_FILE_NAME

    if raw_test_split_path.is_dir():
        try:
            resume_wandb_run(config)
            best_iteration_snapshot_dir = config.best_iteration_snapshot_dir
            remove_path(_get_test_metrics_path(best_iteration_snapshot_dir))

            if not raw_test_input_path.is_dir() or not raw_labels_path.is_file():
                raise FileNotFoundError(
                    f"Expected input/ and labels.csv in {raw_test_split_path} for final test evaluation."
                )

            with tempfile.TemporaryDirectory(prefix="test_eval_data_") as prepared_root:
                prepared_test_dir = Path(prepared_root) / config.dataset
                prepare_test_dataset(
                    source_dir=raw_test_dataset_dir,
                    destination_dir=prepared_test_dir,
                    task_type=config.task_type,
                    input_structure=config.input_structure,
                    label_to_scalar=config.label_to_scalar,
                )
                prepared_labels_path = prepared_test_dir / TEST_SPLIT / LABELS_FILE_NAME

                #TODO should use the step id by importing it
                inference_script_path = best_iteration_snapshot_dir / "model_inference" / "inference.py"
                output_path = config.best_iteration_snapshot_dir / "eval_predictions_test.csv"
                remove_path(output_path)
                remove_path(test_eval_env_dir)
                env_path = test_eval_env_dir / "env"
                create_environment_from_descriptor(
                    config.best_iteration_snapshot_dir / "environment.yml", env_path,
                )
                result = run_inference_script(
                    env_path=env_path,
                    script_path=inference_script_path,
                    input_path=raw_test_input_path,
                    output_path=output_path,
                    #TODO should use the step id by importing it
                    artifacts_dir=best_iteration_snapshot_dir / "model_training" / "training_artifacts",
                )
                if result.returncode != 0:
                    raise AgentScriptFailed(f"Inference on test failed: {str(result)}")
                metrics = compute_metrics(
                    results_file=output_path,
                    labels_path=prepared_labels_path,
                    numeric_label_col=NUMERIC_LABEL_COLUMN_NAME,
                    task_type=config.task_type,
                    evaluation_stage="test",
                )
            if metrics is not None:
                _write_test_metrics(best_iteration_snapshot_dir, metrics)
                if is_wandb_active():
                    wandb.log({f"test/{metric_name}": metric_value for metric_name, metric_value in metrics.items()})
        except Exception as e:
            print("FINAL TEST EVAL FAIL", str(e))
            if is_wandb_active():
                wandb.log(
                    {
                        f"test/{metric_name}": float("nan")
                        for metric_name in get_task_to_metrics_names()[config.task_type]
                    }
                )
        finally:
            remove_path(test_eval_env_dir)
    log_test_inference_duration(time.time() - start)
    print_best_iteration_metrics(config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-dir', type=Path, default=Path('/workspace').resolve(), help='Path to workspace directory')
    parser.add_argument('--test-datasets-dir', type=Path, required=True, help='Path to test datasets root directory')
    args = parser.parse_args()

    run_test_evaluation(
        args.workspace_dir,
        test_datasets_dir=args.test_datasets_dir.resolve(),
    )

if __name__ == "__main__":
    main()
