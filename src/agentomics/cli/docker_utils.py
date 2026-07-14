from __future__ import annotations

import argparse
import os
import shutil
import subprocess


DEFAULT_IMAGE = "biogemt/agentomics:latest"
CONTAINER_PYTHON = "/opt/conda/envs/agentomics-env/bin/python"

def create_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Docker image used for the worker",
    )
    return parser

def run_python_in_docker(
    image: str,
    cpu_only: bool,
    mounts: list[str],
    python_arguments: list[str],
    docker_arguments: list[str] | None = None,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> int:
    if shutil.which("docker") is None:
        raise RuntimeError("Missing required command: docker")

    command = ["docker", "run", "--rm"]
    if docker_arguments:
        command.extend(docker_arguments)
    if not cpu_only:
        command.extend(["--gpus", "all"])
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        command.extend(
            [
                "-e",
                f"HOST_UID={os.getuid()}",
                "-e",
                f"HOST_GID={os.getgid()}",
            ]
        )
    command.extend(["-e", "PYTHONPATH=/repository/src"])
    for mount in mounts:
        command.extend(["--mount", mount])
    command.extend(["--entrypoint", CONTAINER_PYTHON, image])
    command.extend(python_arguments)
    if timeout_seconds is not None:
        command = ["timeout", str(timeout_seconds), *command]
    return subprocess.run(command, check=check).returncode
