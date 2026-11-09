from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentomics.cli.docker_utils import (
    PROXY_ENV_VAR_NAMES,
    build_development_image,
    create_parser,
    docker_environment_arguments,
    run_python_in_docker,
)
from agentomics.cli.run import CONTAINER_CODEX_DIRECTORY

CONTAINER_REQUEST_PATH = Path("/request.json")
CONTAINER_COLLECTED_DATA_DIRECTORY = Path("/collected-data")
ENVIRONMENT_VARIABLE_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    *PROXY_ENV_VAR_NAMES,
    *(name.lower() for name in PROXY_ENV_VAR_NAMES),
)


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("Collect public data for a task: search the web and download the files into a folder.")
    parser.add_argument("--request", type=Path, required=True, help="JSON file describing the data to collect")
    parser.add_argument("--collected-data-dir", type=Path, required=True, help="Folder for the downloaded files; rerun on it to add files")
    parser.add_argument("--provider", required=True, help="LLM provider, for example codex or openrouter")
    parser.add_argument("--model", required=True, help="LLM model for the collector and the reviewer")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    request_path = arguments.request.expanduser().resolve()
    if not request_path.is_file():
        raise FileNotFoundError(f"Request file not found: {request_path}")
    collected_data_directory = arguments.collected_data_dir.expanduser().resolve()
    collected_data_directory.mkdir(parents=True, exist_ok=True)

    if arguments.dev:
        try:
            arguments.image = build_development_image()
        except RuntimeError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    mounts = [
        f"type=bind,src={request_path},dst={CONTAINER_REQUEST_PATH},readonly",
        f"type=bind,src={collected_data_directory},dst={CONTAINER_COLLECTED_DATA_DIRECTORY}",
    ]
    codex_directory = Path.home() / ".codex"
    if codex_directory.is_dir():
        mounts.append(f"type=bind,src={codex_directory.resolve()},dst={CONTAINER_CODEX_DIRECTORY},readonly")

    python_arguments = [
        "-m", "agentomics.agents.data_collection",
        "--request", str(CONTAINER_REQUEST_PATH),
        "--collected-data-dir", str(CONTAINER_COLLECTED_DATA_DIRECTORY),
        "--provider", arguments.provider,
        "--model", arguments.model,
    ]

    return run_python_in_docker(
        image=arguments.image,
        cpu_only=True,
        mounts=mounts,
        python_arguments=python_arguments,
        docker_arguments=docker_environment_arguments(*ENVIRONMENT_VARIABLE_NAMES),
        check=False,
    )


if __name__ == "__main__":
    sys.exit(main())
