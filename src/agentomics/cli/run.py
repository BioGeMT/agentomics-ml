from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from agentomics.cli.docker_utils import (
    create_parser,
    run_python_in_docker,
    validate_docker_gpu_access,
)
from agentomics.cli.inference import run_inference_in_docker
from agentomics.cli.run_arguments import add_run_arguments
from agentomics.datasets.data_contract import TEST_SPLIT
from agentomics.datasets.dataset_preparation import prepare_test_dataset
from agentomics.datasets.datasets_interactive import (
    get_all_datasets_info,
    interactive_dataset_selection,
    print_datasets_table,
)
from agentomics.runtime.read_write_utils import load_config_from_run_dir
from agentomics.utils.agent_id import create_agent_id
from agentomics.utils.config import Config


DEFAULT_DATASETS_DIRECTORY = Path("datasets")
DEFAULT_OUTPUTS_DIRECTORY = Path("outputs")
CONTAINER_CODEX_DIRECTORY = Path("/mnt/codex-host")
CONTAINER_DATASETS_DIRECTORY = Path("/datasets")
CONTAINER_FORK_SOURCE_DIRECTORY = Path("/fork-source")
CONTAINER_WORKSPACE_DIRECTORY = Path("/workspace")
PUBLIC_DATASET_ENTRY_NAMES = (
    "train",
    "train.csv",
    "validation",
    "validation.csv",
    "supplementary",
    "metadata.json",
    "dataset_description.md",
)


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("Run the Agentomics model-development workflow.")
    add_run_arguments(parser)
    return parser


def _resolve_workspace(
    requested_workspace: Path | None,
    agent_id: str,
) -> Path:
    # Default outputs belong to the directory from which the user launched the CLI.
    workspace_directory = (
        requested_workspace or Path.cwd() / DEFAULT_OUTPUTS_DIRECTORY / agent_id
    ).expanduser().resolve()
    workspace_directory.mkdir(parents=True, exist_ok=True)
    return workspace_directory


def _build_container_arguments(arguments: argparse.Namespace) -> list[str]:
    workflow_arguments = [
        "-m",
        "agentomics.runtime.run_workflow",
    ]
    for option, value in (
        ("--model", arguments.model),
        ("--iteration-plan-model", arguments.iteration_plan_model),
        ("--provider", arguments.provider),
        ("--dataset", arguments.dataset),
        ("--task-type", arguments.task_type),
        ("--val-metric", arguments.val_metric),
        ("--iterations", arguments.iterations),
        ("--timeout", arguments.timeout),
        ("--split-timeout", arguments.split_timeout),
        ("--run-python-timeout", arguments.run_python_timeout),
        ("--split-allowed-iterations", arguments.split_allowed_iterations),
        ("--exploration-iterations", arguments.exploration_iterations),
        ("--user-prompt", arguments.user_prompt),
        ("--fork-from-step", arguments.fork_from_step),
        ("--fork-from-iteration", arguments.fork_from_iteration),
        ("--spend-limit", arguments.spend_limit),
        ("--conda-export-mode", arguments.conda_export_mode),
        ("--verbosity", arguments.verbosity),
    ):
        if value is not None:
            workflow_arguments.extend([option, str(value)])
    if arguments.tags is not None:
        workflow_arguments.append("--tags")
        workflow_arguments.extend(arguments.tags)
    for option, enabled in (
        ("--use-provisioning-key", arguments.use_provisioning_key),
        ("--test", arguments.test),
        ("--cpu-only", arguments.cpu_only),
        ("--disable-training-reporting", arguments.disable_training_reporting),
        ("--list-models", arguments.list_models),
        ("--list-datasets", arguments.list_datasets),
        ("--list-metrics", arguments.list_metrics),
    ):
        if enabled:
            workflow_arguments.append(option)
    return workflow_arguments

def _docker_environment_arguments(agent_id: str) -> list[str]:
    docker_arguments: list[str] = []
    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        docker_arguments.extend(["--env-file", str(env_file.resolve())])
    for variable_name in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "PROVISIONING_OPENROUTER_API_KEY", "OLLAMA_BASE_URL",
        "WANDB_API_KEY", "WANDB_PROJECT_NAME", "WANDB_ENTITY",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "CUDA_VISIBLE_DEVICES",
    ):
        if variable_name in os.environ:
            docker_arguments.extend(["-e", variable_name])
    docker_arguments.extend(["-e", f"AGENT_ID={agent_id}"])
    if sys.stdin.isatty() and sys.stdout.isatty():
        docker_arguments.append("-it")
    return docker_arguments


def _resolve_dataset(
    arguments: argparse.Namespace,
    datasets_directory: Path,
) -> Path:
    if arguments.fork_from_run is not None:
        fork_config = load_config_from_run_dir(
            arguments.fork_from_run.expanduser().resolve() / Config.RUN_DIRNAME
        )
        if (
            arguments.dataset is not None
            and arguments.dataset != fork_config.dataset
        ):
            print(
                f"Warning: --dataset is ignored for forked runs. "
                f"Using '{fork_config.dataset}' from the source run config.",
                file=sys.stderr,
            )
        arguments.dataset = fork_config.dataset

    if arguments.dataset is None:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise RuntimeError(
                "Non-interactive runs require --dataset or --fork-from-run"
            )
        arguments.dataset = interactive_dataset_selection(
            get_all_datasets_info(datasets_directory)
        )
    if arguments.dataset is None:
        raise RuntimeError("No dataset selected")

    dataset_directory = (datasets_directory / arguments.dataset).resolve()
    if dataset_directory.parent != datasets_directory:
        raise ValueError(
            f"Dataset must be a direct child of {datasets_directory}: "
            f"{dataset_directory}"
        )
    if not dataset_directory.is_dir():
        raise NotADirectoryError(f"Dataset not found: {dataset_directory}")
    arguments.dataset = dataset_directory.name
    return dataset_directory


def _build_dataset_mounts(
    dataset_directory: Path | None,
) -> list[str]:
    mounts = [f"type=tmpfs,dst={CONTAINER_DATASETS_DIRECTORY}"]
    if dataset_directory is None:
        return mounts

    for entry_name in PUBLIC_DATASET_ENTRY_NAMES:
        source_entry = dataset_directory / entry_name
        if not source_entry.exists():
            continue
        if source_entry.is_symlink():
            raise ValueError(
                f"Dataset entries may not be symlinks: {source_entry}"
            )
        mounts.append(
            f"type=bind,src={source_entry.resolve()},"
            f"dst={CONTAINER_DATASETS_DIRECTORY / dataset_directory.name / entry_name},"
            "readonly"
        )
    return mounts


def _run_agent_in_docker(
    arguments: argparse.Namespace,
    workspace_directory: Path,
    agent_id: str,
    dataset_mounts: list[str],
) -> int:
    mounts = list(dataset_mounts)
    container_arguments = _build_container_arguments(arguments)
    container_arguments.extend(
        ["--datasets-dir", str(CONTAINER_DATASETS_DIRECTORY)]
    )

    mounts.append(
        f"type=bind,src={workspace_directory},dst={CONTAINER_WORKSPACE_DIRECTORY}"
    )
    container_arguments.extend(
        ["--workspace-dir", str(CONTAINER_WORKSPACE_DIRECTORY)]
    )

    if arguments.fork_from_run is not None:
        fork_directory = arguments.fork_from_run.expanduser().resolve()
        if not fork_directory.is_dir():
            raise NotADirectoryError(f"Fork source directory not found: {fork_directory}")
        mounts.append(
            f"type=bind,src={fork_directory},"
            f"dst={CONTAINER_FORK_SOURCE_DIRECTORY},readonly"
        )
        container_arguments.extend(
            ["--fork-from-run", str(CONTAINER_FORK_SOURCE_DIRECTORY)]
        )

    codex_directory = Path.home() / ".codex"
    if codex_directory.is_dir():
        mounts.append(
            f"type=bind,src={codex_directory.resolve()},"
            f"dst={CONTAINER_CODEX_DIRECTORY},readonly"
        )

    docker_arguments = _docker_environment_arguments(agent_id)
    if arguments.provider == "ollama":
        docker_arguments.extend(["--network", "host"])
    # A non-zero exit is an expected outcome (for example, no best-iteration
    # snapshot); the workflow already reports it, so return the code instead of
    # raising and let the caller skip post-run steps cleanly.
    return run_python_in_docker(
        image=arguments.image,
        cpu_only=arguments.cpu_only,
        mounts=mounts,
        python_arguments=container_arguments,
        docker_arguments=docker_arguments,
        check=False,
    )


def _run_inference_on_test_input(
    image: str,
    cpu_only: bool,
    workspace_directory: Path,
    test_input: Path,
) -> None:
    print(f"Evaluating the best iteration on {test_input.name}: {test_input}")
    run_inference_in_docker(
        argparse.Namespace(
            image=image,
            agent_dir=workspace_directory,
            input_path=test_input,
            output=(
                workspace_directory
                / Config.BEST_ITERATION_SNAPSHOT_DIRNAME
                / f"eval_predictions_{test_input.name}.csv"
            ),
            label_col=None,
            iteration_dir=Path(Config.BEST_ITERATION_SNAPSHOT_DIRNAME),
            all_iterations=False,
            cpu_only=cpu_only,
            wandb_prefix=test_input.name,
        )
    )


def _run_test_evaluation_in_docker(
    image: str,
    cpu_only: bool,
    workspace_directory: Path,
    dataset_directory: Path,
) -> None:
    test_directories = sorted(
        path
        for path in dataset_directory.iterdir()
        if path.is_dir() and path.name.startswith(TEST_SPLIT)
    )
    if test_directories:
        for test_directory in test_directories:
            try:
                _run_inference_on_test_input(
                    image,
                    cpu_only,
                    workspace_directory,
                    test_directory,
                )
            except subprocess.CalledProcessError as error:
                print(
                    f"Warning: Evaluation failed for {test_directory.name}; "
                    f"continuing: {error}",
                    file=sys.stderr,
                )
        return

    test_csv = dataset_directory / "test.csv"
    if not test_csv.is_file():
        print(f"No test split found in {dataset_directory}; skipping test evaluation.")
        return

    metadata_path = (
        workspace_directory
        / "prepared_datasets"
        / dataset_directory.name
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary_directory:
        prepared_directory = Path(temporary_directory) / dataset_directory.name
        prepare_test_dataset(
            source_dir=dataset_directory,
            destination_dir=prepared_directory,
            task_type=metadata["task_type"],
            input_structure=metadata["input_structure"],
            label_to_scalar=metadata.get("label_to_scalar"),
            label_column=metadata.get("label_column"),
            id_column=metadata.get("id_column"),
        )
        _run_inference_on_test_input(
            image,
            cpu_only,
            workspace_directory,
            prepared_directory / "test",
        )


def _run_reporting_in_docker(
    image: str,
    workspace_directory: Path,
    agent_id: str,
) -> None:
    run_python_in_docker(
        image=image,
        cpu_only=True,
        mounts=[
            f"type=bind,src={workspace_directory},"
            f"dst={CONTAINER_WORKSPACE_DIRECTORY}"
        ],
        python_arguments=[
            "-m",
            "agentomics.runtime.report_workflow",
            "--workspace-dir",
            str(CONTAINER_WORKSPACE_DIRECTORY),
            "--agent-id",
            agent_id,
        ],
    )


def _is_agent_run(arguments: argparse.Namespace) -> bool:
    # TODO: Replace this implicit check with explicit command modes.
    return not (
        arguments.test
        or arguments.list_models
        or arguments.list_datasets
        or arguments.list_metrics
    )

def main() -> int:
    arguments = build_parser().parse_args()
    datasets_directory = (
        arguments.datasets_dir or Path.cwd() / DEFAULT_DATASETS_DIRECTORY
    ).expanduser().resolve()
    if not datasets_directory.is_dir():
        raise NotADirectoryError(
            f"Datasets directory not found: {datasets_directory}"
        )

    if arguments.list_datasets:
        print_datasets_table(get_all_datasets_info(datasets_directory))
        return 0

    if not arguments.cpu_only:
        try:
            validate_docker_gpu_access(arguments.image)
        except RuntimeError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    dataset_directory = (
        _resolve_dataset(arguments, datasets_directory)
        if _is_agent_run(arguments)
        else None
    )
    agent_id = create_agent_id()
    workspace_directory = _resolve_workspace(arguments.workspace_dir, agent_id)
    dataset_mounts = _build_dataset_mounts(dataset_directory)
    exit_code = _run_agent_in_docker(
        arguments,
        workspace_directory,
        agent_id,
        dataset_mounts,
    )
    if _is_agent_run(arguments) and exit_code == 0:
        try:
            _run_test_evaluation_in_docker(
                image=arguments.image,
                cpu_only=arguments.cpu_only,
                workspace_directory=workspace_directory,
                dataset_directory=dataset_directory,
            )
        finally:
            _run_reporting_in_docker(
                arguments.image,
                workspace_directory,
                agent_id,
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
