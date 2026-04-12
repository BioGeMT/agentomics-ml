from __future__ import annotations

import os
import subprocess
from pathlib import Path

from runtime.filesystem import remove_path


def get_shared_conda_root(config) -> Path:
    return config.shared_dir / ".conda"


def get_shared_environment_path(config) -> Path:
    return get_shared_conda_root(config) / "envs" / f"{config.agent_id}_env"


def get_snapshot_environment_path(config) -> Path:
    return config.snapshot_dir / ".conda" / "envs" / f"{config.agent_id}_env"


def get_iteration_environment_descriptor_path(iteration_dir: Path) -> Path:
    return iteration_dir / "runtime_info" / "environment.yml"


def get_iteration_test_environment_path(agent_dir: Path, iteration_dir: Path) -> Path:
    return agent_dir / "_iteration_test_envs" / iteration_dir.name / "env"


def remove_environment(env_path: Path) -> None:
    remove_path(env_path)


def materialize_environment_from_descriptor(descriptor_path: Path, env_path: Path, working_dir: Path) -> None:
    if not descriptor_path.exists():
        return

    remove_environment(env_path)
    env_path.parent.mkdir(parents=True, exist_ok=True)

    start_env_pkg = os.getenv("START_ENV_PKG")
    if start_env_pkg:
        subprocess.run(
            f"mkdir -p {env_path} && tar -xf {start_env_pkg} -C {env_path} && source {env_path}/bin/activate && conda-unpack",
            shell=True,
            executable="/bin/bash",
            check=True,
            cwd=working_dir,
        )

    subprocess.run(
        [
            "conda",
            "env",
            "update",
            "-p",
            str(env_path),
            "-f",
            str(descriptor_path),
            "-q",
        ],
        check=True,
        cwd=working_dir,
    )
def export_shared_environment_descriptor(config) -> None:
    conda_env = get_shared_environment_path(config)
    export_environment_descriptor_to_path(
        env_path=conda_env,
        descriptor_path=config.shared_dir / "environment.yml",
    )

def export_environment_descriptor_to_path(env_path: Path, descriptor_path: Path) -> None:
    if not env_path.exists():
        return
    subprocess.run(
        [
            "conda",
            "env",
            "export",
            "-p",
            str(env_path),
            "-f",
            str(descriptor_path),
        ],
        check=True,
        capture_output=True,
    )
