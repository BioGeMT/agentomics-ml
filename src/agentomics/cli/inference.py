from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentomics.cli.docker_utils import (
    create_parser,
    run_python_in_docker,
)

from agentomics.runtime.inference_workflow import resolve_inference_paths
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
        "--all-iterations", action="store_true",
        help="Run every archived iteration",
    )
    parser.add_argument(
        "--remove-conda-env", action="store_true",
        help="Remove the model environment afterward",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU access")
    return parser

def _run_in_docker(arguments: argparse.Namespace) -> None:
    resolve_inference_paths(arguments)
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
    if arguments.label_col:
        python_arguments.extend(["--label-col", arguments.label_col])
    if arguments.all_iterations:
        python_arguments.append("--all-iterations")
    if arguments.remove_conda_env:
        python_arguments.append("--remove-conda-env")
    if arguments.cpu_only:
        python_arguments.append("--cpu-only")

    run_python_in_docker(
        image=arguments.image,
        cpu_only=arguments.cpu_only,
        mounts=[
            f"type=bind,src={arguments.agent_dir},dst=/agent",
            f"type=bind,src={arguments.input_path},dst={container_input},readonly",
            f"type=bind,src={arguments.output.parent},dst=/inference-output",
        ],
        python_arguments=python_arguments,
    )

def main() -> int:
    arguments = build_parser().parse_args()
    _run_in_docker(arguments)
    return 0

if __name__ == "__main__":
    sys.exit(main())
