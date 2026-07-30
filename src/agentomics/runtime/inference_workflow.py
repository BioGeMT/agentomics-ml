from __future__ import annotations

import os
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from agentomics.datasets.csv_converter import convert_inference_csv
from agentomics.datasets.data_contract import INPUT_DIR_NAME, LABELS_FILE_NAME
from agentomics.runtime.conda_utils import restore_iteration_environment
from agentomics.runtime.evaluate import evaluate_predictions
from agentomics.runtime.filesystem import (
    resolve_inference_paths,
    resolve_iteration_root,
    restore_host_ownership,
)
from agentomics.runtime.inference_runner import run_inference_script
from agentomics.utils.config import Config

def prepare_inference_input(
    input_path: Path,
    output_split_dir: Path,
    label_column: str | None = None,
    id_column: str | None = None,
) -> Path:
    input_path = Path(input_path)
    if input_path.is_dir():
        if not (input_path / INPUT_DIR_NAME).is_dir():
            raise FileNotFoundError(
                f"--input folder must contain {INPUT_DIR_NAME}/: {input_path}"
            )
        return input_path

    output_split_dir = Path(output_split_dir)
    convert_inference_csv(
        csv_path=input_path,
        output_split_dir=output_split_dir,
        label_column=label_column,
        id_column=id_column,
    )
    return output_split_dir

def _compute_metrics(agent_dir: Path, input_split: Path, output_path: Path, wandb_prefix: str | None) -> Path | None:
    labels_path = input_split / LABELS_FILE_NAME
    if not labels_path.is_file():
        return None

    metrics_path = output_path.with_suffix(".metrics.json")
    print("Computing metrics...")
    evaluate_predictions(agent_dir, output_path, labels_path, metrics_path, wandb_prefix)
    return metrics_path

def _run_single_inference(arguments: Namespace) -> None:
    iteration_root = resolve_iteration_root(arguments.agent_dir, arguments.iteration_dir)
    inference_script = iteration_root / "model_inference" / "inference.py"
    if not inference_script.is_file():
        raise FileNotFoundError(f"inference.py not found at: {inference_script}")

    artifacts_path = iteration_root / "model_training" / "training_artifacts"
    print(f"Using iteration directory: {arguments.iteration_dir}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        environment_path = restore_iteration_environment(
            iteration_root,
            temporary_root / "model_env",
        )
        print(f"Using temporary model env: {environment_path}")
        input_split = prepare_inference_input(
            input_path=arguments.input_path,
            output_split_dir=temporary_root / "input",
            label_column=arguments.label_col,
        )
        environment = os.environ.copy()
        if arguments.cpu_only:
            print("Running in CPU-only mode")
            environment["CUDA_VISIBLE_DEVICES"] = ""

        print("Running inference")
        run_inference_script(
            env_path=environment_path,
            script_path=inference_script,
            input_path=input_split / INPUT_DIR_NAME,
            output_path=arguments.output,
            artifacts_dir=artifacts_path,
            check=True,
            capture_output=False,
            environment=environment,
        )
        if not arguments.output.is_file():
            raise FileNotFoundError(
                f"Inference completed without creating the prediction output: {arguments.output}"
            )
        print("Inference done")
        metrics_path = _compute_metrics(arguments.agent_dir, input_split, arguments.output, arguments.wandb_prefix)
        restore_host_ownership(arguments.output)
        if metrics_path is not None:
            restore_host_ownership(metrics_path)

def _run_all_iterations(arguments: Namespace) -> None:
    run_directory = arguments.agent_dir / Config.RUN_DIRNAME
    if not run_directory.is_dir():
        raise FileNotFoundError(f"--all-iterations requires {run_directory}, not found")

    iteration_directories = sorted(
        (
            path
            for path in run_directory.glob(f"{Config.ITERATION_DIR_PREFIX}*")
            if path.is_dir() and path.name.removeprefix(Config.ITERATION_DIR_PREFIX).isdigit()
        ),
        key=lambda path: int(path.name.removeprefix(Config.ITERATION_DIR_PREFIX)),
    )
    if not iteration_directories:
        raise FileNotFoundError(
            f"No {Config.ITERATION_DIR_PREFIX}* directories found under {run_directory}"
        )

    print(f"Evaluating {len(iteration_directories)} archived iteration(s) on {arguments.input_path}")
    successful_iterations = 0
    failed_iterations = 0
    for iteration_directory in iteration_directories:
        iteration_name = iteration_directory.name
        print(f"=== {iteration_name} ===")
        try:
            iteration_arguments = Namespace(**vars(arguments))
            iteration_arguments.iteration_dir = Path(Config.RUN_DIRNAME) / iteration_name
            iteration_arguments.output = arguments.output.parent / f"{iteration_name}_predictions.csv"
            _run_single_inference(iteration_arguments)
            successful_iterations += 1
        except Exception as error:
            failed_iterations += 1
            print(f"Warning: Inference failed for {iteration_name}; continuing: {error}", file=sys.stderr)
    print(
        f"All iterations done: {successful_iterations} succeeded, {failed_iterations} failed. "
        f"Predictions written to {arguments.output.parent}/<iteration>_predictions.csv"
    )
    if successful_iterations == 0:
        raise RuntimeError("Inference failed for every archived iteration")

def run_inference_workflow(arguments: Namespace) -> None:
    if shutil.which("conda") is None:
        raise RuntimeError("Missing required command: conda")
    resolve_inference_paths(arguments)

    if arguments.all_iterations:
        _run_all_iterations(arguments)
    else:
        _run_single_inference(arguments)


def main() -> int:
    from agentomics.cli.inference import build_parser

    run_inference_workflow(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
