from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentomics.cli.docker_utils import (
    build_development_image,
    create_parser,
    run_python_in_docker,
)
from agentomics.runtime.training_workflow import resolve_training_paths
from agentomics.utils.config import Config

def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("Retrain an Agentomics model on a compatible dataset.")
    parser.add_argument(
        "--agent-dir", type=Path, required=True,
        help="Path to the agent output directory",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Dataset directory containing train and validation splits",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, required=True,
        help="Directory where training artifacts will be written",
    )
    parser.add_argument("--label-col", help="Label column for CSV-form datasets")
    parser.add_argument(
        "--iteration-dir",
        type=Path,
        default=Path(Config.BEST_ITERATION_SNAPSHOT_DIRNAME),
        help="Iteration directory relative to --agent-dir",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU access")
    return parser

def _run_in_docker(arguments: argparse.Namespace) -> None:
    python_arguments = [
        "-m",
        "agentomics.runtime.training_workflow",
        "--agent-dir",
        "/agent",
        "--dataset-dir",
        "/training-data",
        "--artifacts-dir",
        f"/training-output/{arguments.artifacts_dir.name}",
        "--iteration-dir",
        str(arguments.iteration_dir),
    ]
    if arguments.label_col:
        python_arguments.extend(["--label-col", arguments.label_col])
    if arguments.cpu_only:
        python_arguments.append("--cpu-only")

    run_python_in_docker(
        image=arguments.image,
        cpu_only=arguments.cpu_only,
        mounts=[
            f"type=bind,src={arguments.agent_dir},dst=/agent",
            f"type=bind,src={arguments.dataset_dir},dst=/training-data,readonly",
            f"type=bind,src={arguments.artifacts_dir.parent},dst=/training-output",
        ],
        python_arguments=python_arguments,
    )

def main() -> int:
    arguments = build_parser().parse_args()
    resolve_training_paths(arguments)
    if arguments.dev:
        try:
            arguments.image = build_development_image()
        except RuntimeError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
    _run_in_docker(arguments)
    return 0

if __name__ == "__main__":
    sys.exit(main())
