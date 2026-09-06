from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentomics.cli.docker_utils import (
    build_development_image,
    create_parser,
    docker_environment_arguments,
    run_python_in_docker,
)

from agentomics.runtime.filesystem import resolve_inference_paths
from agentomics.utils.config import Config

def build_parser() -> argparse.ArgumentParser:
    parser = create_parser(
        "Run a trained Agentomics model on any compatible input data."
    )
    parser.add_argument(
        "--agent-dir", type=Path, required=True,
        help="Path to the agent output directory",
    )
    parser.add_argument(
        "--input", dest="input_path", type=Path, required=True,
        help="Input split directory or CSV file",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Prediction output CSV"
    )
    parser.add_argument("--label-col", help="Label column when the input is a CSV file")
    parser.add_argument(
        "--iteration-dir",
        type=Path,
        default=Path(Config.BEST_ITERATION_SNAPSHOT_DIRNAME),
        help="Iteration directory relative to --agent-dir",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path,
        help=(
            "Use these model artifacts instead of the selected iteration's original artifacts "
            "(cannot be combined with --all-iterations)"
        ),
    )
    parser.add_argument(
        "--all-iterations", action="store_true",
        help="Run every archived iteration",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU access")
    parser.add_argument("--wandb-prefix", help="Enable W&B logging with this prefix; valid WANDB_API_KEY, WANDB_PROJECT_NAME, and WANDB_ENTITY are also required")
    return parser

def run_inference_in_docker(arguments: argparse.Namespace) -> int:
    resolve_inference_paths(arguments)
    return _run_inference_with_resolved_paths(arguments)


def _run_inference_with_resolved_paths(arguments: argparse.Namespace) -> int:
    container_input = "/inference-input"
    python_arguments = [
        "-m",
        "agentomics.runtime.inference_workflow",
        "--agent-dir",
        "/agent",
        "--input",
        container_input,
        "--output",
        f"/inference-output/{arguments.output.name}",
        "--iteration-dir",
        str(arguments.iteration_dir),
    ]
    mounts = [
        f"type=bind,src={arguments.agent_dir},dst=/agent,readonly",
        f"type=bind,src={arguments.input_path},dst={container_input},readonly",
        f"type=bind,src={arguments.output.parent},dst=/inference-output",
    ]
    if arguments.artifacts_dir is not None:
        mounts.append(
            f"type=bind,src={arguments.artifacts_dir},dst=/inference-artifacts,readonly"
        )
        python_arguments.extend(["--artifacts-dir", "/inference-artifacts"])
    if arguments.label_col:
        python_arguments.extend(["--label-col", arguments.label_col])
    if arguments.all_iterations:
        python_arguments.append("--all-iterations")
    if arguments.cpu_only:
        python_arguments.append("--cpu-only")
    if arguments.wandb_prefix:
        python_arguments.extend(["--wandb-prefix", arguments.wandb_prefix])

    return run_python_in_docker(
        image=arguments.image,
        cpu_only=arguments.cpu_only,
        mounts=mounts,
        python_arguments=python_arguments,
        docker_arguments=docker_environment_arguments(
            "WANDB_API_KEY",
            "WANDB_PROJECT_NAME",
            "WANDB_ENTITY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ),
        check=False,
    )

def main() -> int:
    arguments = build_parser().parse_args()
    resolve_inference_paths(arguments)
    if arguments.dev:
        try:
            arguments.image = build_development_image()
        except RuntimeError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
    return _run_inference_with_resolved_paths(arguments)

if __name__ == "__main__":
    sys.exit(main())
