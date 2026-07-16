from __future__ import annotations

import os
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from agentomics.datasets.data_contract import TRAIN_SPLIT, VALIDATION_SPLIT
from agentomics.datasets.dataset_preparation import prepare_dataset
from agentomics.runtime.conda_utils import (
    restore_iteration_environment,
    run_python_in_environment,
)
from agentomics.runtime.filesystem import (
    require_empty_directory,
    resolve_existing_directory,
    resolve_iteration_root,
    resolve_output_path,
    restore_host_ownership,
)
from agentomics.runtime.read_write_utils import load_config_from_run_dir
from agentomics.utils.config import Config


def resolve_training_paths(arguments: Namespace) -> None:
    arguments.agent_dir = resolve_existing_directory(arguments.agent_dir)
    arguments.dataset_dir = resolve_existing_directory(arguments.dataset_dir)
    arguments.artifacts_dir = resolve_output_path(arguments.artifacts_dir)
    require_empty_directory(arguments.artifacts_dir)


def _print_training_summary(artifacts_directory: Path) -> None:
    print("\n========== Training Summary ==========")
    print(f"Artifacts created in: {artifacts_directory}")
    for artifact in sorted(artifacts_directory.iterdir()):
        print(f"- {artifact.name}")
    metadata_path = artifacts_directory / "metadata.json"
    if metadata_path.is_file():
        print("\nMetadata:")
        print(metadata_path.read_text(encoding="utf-8"))
    print("======================================")


def run_training_workflow(arguments: Namespace) -> None:
    if shutil.which("conda") is None:
        raise RuntimeError("Missing required command: conda")
    resolve_training_paths(arguments)

    iteration_root = resolve_iteration_root(
        arguments.agent_dir,
        arguments.iteration_dir,
    )
    training_script = iteration_root / "model_training" / "train.py"
    if not training_script.is_file():
        raise FileNotFoundError(f"train.py not found at: {training_script}")

    print(f"Using iteration directory: {arguments.iteration_dir}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        environment_path = restore_iteration_environment(
            iteration_root,
            temporary_root / "model_env",
        )
        print(f"Using temporary model env: {environment_path}")
        prepared_directory = temporary_root / "prepared"
        config = load_config_from_run_dir(
            arguments.agent_dir / Config.RUN_DIRNAME,
            missing_ok=True,
        )
        if config is None:
            raise FileNotFoundError(
                f"Could not load run config under "
                f"{arguments.agent_dir / Config.RUN_DIRNAME}"
            )
        prepare_dataset(
            source_dir=arguments.dataset_dir,
            destination_dir=prepared_directory,
            task_type=config.task_type,
            label_to_scalar=config.label_to_scalar,
            label_column=arguments.label_col,
        )
        train_data = prepared_directory / TRAIN_SPLIT
        validation_data = prepared_directory / VALIDATION_SPLIT
        if not validation_data.is_dir():
            raise FileNotFoundError(
                f"Prepared validation split not found: {validation_data}"
            )
        environment = os.environ.copy()
        if arguments.cpu_only:
            print("Training in CPU-only mode")
            environment["CUDA_VISIBLE_DEVICES"] = ""

        print("Running training...")
        run_python_in_environment(
            environment_path,
            training_script,
            [
                "--train-data", str(train_data),
                "--validation-data", str(validation_data),
                "--artifacts-dir", str(arguments.artifacts_dir),
            ],
            capture_output=False,
            check=True,
            environment=environment,
        )
        print("Training done")

    if not arguments.artifacts_dir.is_dir() or not any(
        arguments.artifacts_dir.iterdir()
    ):
        raise FileNotFoundError(
            f"Training completed without creating artifacts in "
            f"{arguments.artifacts_dir}"
        )
    _print_training_summary(arguments.artifacts_dir)
    restore_host_ownership(arguments.artifacts_dir)


def main() -> int:
    from agentomics.cli.retrain import build_parser

    run_training_workflow(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
