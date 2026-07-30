from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from agentomics.utils.versioning import get_version


DEFAULT_IMAGE = f"biogemt/agentomics:{get_version()}"
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

def docker_environment_arguments(*variable_names: str) -> list[str]:
    docker_arguments: list[str] = []
    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        docker_arguments.extend(["--env-file", str(env_file.resolve())])
    for variable_name in variable_names:
        if variable_name in os.environ:
            docker_arguments.extend(["-e", variable_name])
    return docker_arguments

def validate_docker_gpu_access(image: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Missing required command: docker")

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "/bin/true",
            image,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "GPU is enabled by default, but Docker could not allocate a GPU. "
            "Use --cpu-only to run without GPU access."
        )

def run_python_in_docker(
    image: str,
    cpu_only: bool,
    mounts: list[str],
    python_arguments: list[str],
    docker_arguments: list[str] | None = None,
    check: bool = True,
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
    return subprocess.run(command, check=check).returncode
