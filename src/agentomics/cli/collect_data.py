from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentomics.cli.docker_utils import (
    PROVIDER_API_KEY_ENV_VAR_NAMES,
    PROXY_ENV_VAR_NAMES,
    codex_credentials_mounts,
    create_parser,
    docker_environment_arguments,
    run_python_in_docker,
    use_development_image_if_requested,
)
from agentomics.runtime.filesystem import resolve_existing_path

CONTAINER_REQUEST_PATH = Path("/request.json")
CONTAINER_COLLECTED_DATA_DIRECTORY = Path("/collected-data")


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("Collect public data for a task: search the web and download the files into a folder.")
    parser.add_argument("--request", type=Path, required=True, help="JSON file describing the data to collect")
    parser.add_argument("--collected-data-dir", type=Path, required=True, help="Folder for the downloaded files; rerun on it to add files")
    parser.add_argument("--provider", required=True, help="LLM provider, for example codex or openrouter")
    parser.add_argument("--model", required=True, help="LLM model for the collector and the reviewer")
    return parser


def build_mounts(request_path: Path, collected_data_directory: Path) -> list[str]:
    return [
        f"type=bind,src={request_path},dst={CONTAINER_REQUEST_PATH},readonly",
        f"type=bind,src={collected_data_directory},dst={CONTAINER_COLLECTED_DATA_DIRECTORY}",
    ]


def build_container_arguments(arguments: argparse.Namespace) -> list[str]:
    return [
        "-m", "agentomics.runtime.collect_data_workflow",
        "--request", str(CONTAINER_REQUEST_PATH),
        "--collected-data-dir", str(CONTAINER_COLLECTED_DATA_DIRECTORY),
        "--provider", arguments.provider,
        "--model", arguments.model,
    ]


def main() -> int:
    arguments = build_parser().parse_args()
    request_path = resolve_existing_path(arguments.request)
    if not use_development_image_if_requested(arguments):
        return 1
    collected_data_directory = arguments.collected_data_dir.expanduser().resolve()
    collected_data_directory.mkdir(parents=True, exist_ok=True)

    return run_python_in_docker(
        image=arguments.image,
        cpu_only=True,
        mounts=[*build_mounts(request_path, collected_data_directory), *codex_credentials_mounts()],
        python_arguments=build_container_arguments(arguments),
        docker_arguments=docker_environment_arguments(
            *PROVIDER_API_KEY_ENV_VAR_NAMES,
            *PROXY_ENV_VAR_NAMES,
            *(name.lower() for name in PROXY_ENV_VAR_NAMES),
        ),
        check=False,
    )


if __name__ == "__main__":
    sys.exit(main())
