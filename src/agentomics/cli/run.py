from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agentomics.cli.docker_utils import (
    DEFAULT_IMAGE,
    run_python_in_docker,
)
from agentomics.cli.run_arguments import create_run_parser
from agentomics.utils.agent_id import create_agent_id


DEFAULT_DATASETS_DIRECTORY = Path("datasets")
DEFAULT_OUTPUTS_DIRECTORY = Path("outputs")
CONTAINER_CODEX_DIRECTORY = Path("/mnt/codex-host")
CONTAINER_DATASETS_DIRECTORY = Path("/datasets")
CONTAINER_FORK_SOURCE_DIRECTORY = Path("/fork-source")
CONTAINER_WORKSPACE_DIRECTORY = Path("/workspace")


def build_parser() -> argparse.ArgumentParser:
    parser = create_run_parser()
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Docker image used for the worker",
    )
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

def _run_agent_in_docker(
    arguments: argparse.Namespace,
    workspace_directory: Path,
    agent_id: str,
) -> int:
    datasets_directory = (
        arguments.datasets_dir or Path.cwd() / DEFAULT_DATASETS_DIRECTORY
    ).expanduser().resolve()
    if not datasets_directory.is_dir():
        raise NotADirectoryError(f"Datasets directory not found: {datasets_directory}")

    mounts = [
        f"type=bind,src={datasets_directory},"
        f"dst={CONTAINER_DATASETS_DIRECTORY},readonly"
    ]
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
    run_python_in_docker(
        image=arguments.image,
        cpu_only=arguments.cpu_only,
        mounts=mounts,
        python_arguments=container_arguments,
        docker_arguments=docker_arguments,
    )
    return 0


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
    #TODO can be simplified? have explicit list/module 
    return not (
        arguments.test
        or arguments.list_models
        or arguments.list_datasets
        or arguments.list_metrics
    )

def main() -> int:
    arguments = build_parser().parse_args()
    agent_id = create_agent_id()
    workspace_directory = _resolve_workspace(arguments.workspace_dir, agent_id)
    exit_code = _run_agent_in_docker(arguments, workspace_directory, agent_id)
    if _is_agent_run(arguments):
        _run_reporting_in_docker(arguments.image, workspace_directory, agent_id)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
