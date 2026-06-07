from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path

import wandb

from runtime.evaluate_result import get_metrics
from run_logging.logging_helpers import define_serial_metrics, log_serial_metrics
from run_logging.wandb_setup import resume_wandb_run
from runtime.conda_utils import (
    create_environment_from_descriptor,
    get_iteration_environment_descriptor_path,
    get_iteration_test_environment_path,
    remove_environment,
)
from runtime.filesystem import remove_path
from runtime.inference_runner import run_inference_script
from runtime.read_write_utils import get_archived_iterations, load_config_from_run_dir_and_reroot, load_dataset_metadata
from utils.config import Config
from datasets.data_contract import INPUT_DIR_NAME, LABELS_FILE_NAME, TEST_SPLIT

ITERATION_TEST_ENVS_DIRNAME = "_iteration_test_envs"
STEALTH_TEST_METRICS_FILENAME = "stealth_test_metrics.json"
STEALTH_TEST_METRIC_PREFIX = "stealth_test"
STEALTH_TEST_STEP_METRIC = "stealth_test/iteration"

def evaluate_stealth_test_history(agent_dir: Path, prepared_test_sets_dir: Path) -> None:
    config = load_config_from_run_dir_and_reroot(agent_dir / Config.RUN_DIRNAME)
    metadata = load_dataset_metadata(config)
    test_split_path = prepared_test_sets_dir / config.dataset / TEST_SPLIT
    test_input_path = test_split_path / INPUT_DIR_NAME
    labels_path = test_split_path / LABELS_FILE_NAME
    if not test_input_path.is_dir() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Expected input/ and labels.csv in {test_split_path} for iteration test evaluation."
        )

    run = resume_wandb_run(config, dir=config.extras_dir / "iteration_test_logs")
    define_serial_metrics(STEALTH_TEST_METRIC_PREFIX, config.task_type, step_metric=STEALTH_TEST_STEP_METRIC)
    try:
        results = _evaluate_iterations(
            agent_dir=agent_dir,
            config=config,
            metadata=metadata,
            test_input_path=test_input_path,
            labels_path=labels_path,
        )
    finally:
        if run is not None:
            wandb.finish()

    remove_path(agent_dir / ITERATION_TEST_ENVS_DIRNAME)

    results_path = config.extras_dir / STEALTH_TEST_METRICS_FILENAME
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

def _evaluate_iterations(
    agent_dir: Path,
    config: Config,
    metadata: dict[str, object],
    test_input_path: Path,
    labels_path: Path,
) -> list[dict]:
    iteration_dirs = [
        (iteration, config.iteration_dir(iteration))
        for iteration in get_archived_iterations(config, only_successful=True)
    ]
    results = []

    with tempfile.TemporaryDirectory(prefix="iteration_test_eval_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for iteration, iteration_dir in iteration_dirs:
            print(f"Evaluating iteration {iteration} on held-out test set")
            result = _evaluate_single_iteration(
                agent_dir=agent_dir,
                iteration=iteration,
                iteration_dir=iteration_dir,
                config=config,
                metadata=metadata,
                test_input_path=test_input_path,
                labels_path=labels_path,
                temp_dir=temp_dir_path,
            )
            results.append(result)
            log_serial_metrics(
                prefix=STEALTH_TEST_METRIC_PREFIX,
                task_type=config.task_type,
                metrics=result["metrics"] or None,
                iteration=result["iteration"],
                step_metric=STEALTH_TEST_STEP_METRIC,
            )
    return results

def _evaluate_single_iteration(
    agent_dir: Path,
    iteration: int,
    iteration_dir: Path,
    config: Config,
    metadata: dict[str, object],
    test_input_path: Path,
    labels_path: Path,
    temp_dir: Path,
) -> dict:
    #TODO should use the step id by importing it
    inference_script_path = iteration_dir / "model_inference" / "inference.py"
    if not inference_script_path.exists():
        return {"iteration": iteration, "status": "skipped", "metrics": {}, "error": f"Missing inference.py in {iteration_dir}"}

    output_path = temp_dir / f"{iteration_dir.name}_test_predictions.csv"
    env_path = get_iteration_test_environment_path(agent_dir, iteration_dir)
    try:
        create_environment_from_descriptor(get_iteration_environment_descriptor_path(iteration_dir), env_path)
        run_inference_script(
            env_path=env_path,
            script_path=inference_script_path,
            input_path=test_input_path,
            output_path=output_path,
            artifacts_dir=iteration_dir / "model_training" / "training_artifacts",
            check=True,
        )
        metrics = get_metrics(
            results_file=output_path,
            test_file=labels_path,
            numeric_label_col=metadata["numeric_label_col"],
            task_type=metadata["task_type"],
        )
        return {"iteration": iteration, "status": "success", "metrics": metrics, "error": None}
    except Exception:
        return {"iteration": iteration, "status": "failed", "metrics": {}, "error": traceback.format_exc()}
    finally:
        remove_environment(env_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all successful archived iterations on the held-out test set.")
    parser.add_argument("--agent-dir", type=Path, required=True, help="Path to exported agent output directory")
    parser.add_argument("--prepared-test-sets-dir", type=Path, required=True, help="Path to prepared test sets root directory")
    args = parser.parse_args()
    evaluate_stealth_test_history(
        args.agent_dir.resolve(),
        prepared_test_sets_dir=args.prepared_test_sets_dir.resolve(),
    )

if __name__ == "__main__":
    main()
