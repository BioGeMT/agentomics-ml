from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from agentomics.cli.docker_utils import (
    build_development_image,
    run_python_in_docker,
    validate_docker_gpu_access,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run unit/integration tests in Docker, then end-to-end tests on the host.",
        allow_abbrev=False,
    )
    parser.add_argument("--image", help="Reuse an already-built Docker image")
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU access and GPU tests")
    arguments = parser.parse_args(argv)

    image = arguments.image or build_development_image()
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(f"Docker image {image!r} is unavailable: {result.stderr.strip()}")
    if not arguments.cpu_only:
        validate_docker_gpu_access(image)

    container_name = f"agentomics-tests-{uuid4().hex}"
    docker_arguments = ["--name", container_name, "--pull", "never"]
    if arguments.cpu_only:
        docker_arguments.extend(["-e", "CUDA_VISIBLE_DEVICES="])
    try:
        container_exit = run_python_in_docker(
            image=image,
            cpu_only=arguments.cpu_only,
            mounts=[],
            python_arguments=["-m", "pytest", "tests/unit", "tests/integration"],
            docker_arguments=docker_arguments,
            check=False,
        )
    finally:
        # --rm does not cover interruption of the Docker client.
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    host_exit = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/end_to_end"],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ | {
            "AGENTOMICS_TEST_IMAGE": image,
            "AGENTOMICS_TEST_CPU_ONLY": "1" if arguments.cpu_only else "0",
        },
    ).returncode
    return int(bool(container_exit or host_exit))


if __name__ == "__main__":
    sys.exit(main())
