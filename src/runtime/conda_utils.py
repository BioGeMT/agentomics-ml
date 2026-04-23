from __future__ import annotations

import subprocess
from pathlib import Path

from runtime.filesystem import remove_path
from utils.config import Config


def get_shared_conda_root(config: Config) -> Path:
    return config.shared_dir / ".conda"


def get_shared_environment_path(config: Config) -> Path:
    return get_shared_conda_root(config) / "envs" / f"{config.agent_id}_env"


def get_best_iteration_snapshot_environment_path(config: Config) -> Path:
    return config.best_iteration_snapshot_dir / ".conda" / "envs" / f"{config.agent_id}_env"


def get_iteration_environment_descriptor_path(iteration_dir: Path) -> Path:
    return iteration_dir / "runtime_info" / "environment.yml"


def get_iteration_test_environment_path(agent_dir: Path, iteration_dir: Path) -> Path:
    return agent_dir / "_iteration_test_envs" / iteration_dir.name / "env"


def remove_environment(env_path: Path) -> None:
    remove_path(env_path)


def create_environment_from_descriptor(descriptor_path: Path, env_path: Path) -> None:
    if not descriptor_path.exists():
        raise FileNotFoundError(f"Missing environment descriptor at {descriptor_path}.")

    remove_environment(env_path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["conda", "env", "create", "-p", str(env_path), "-f", str(descriptor_path), "-q"],
        check=True,
    )

def update_environment_from_descriptor(descriptor_path: Path, env_path: Path) -> None:
    if not descriptor_path.exists():
        raise FileNotFoundError(f"Missing environment descriptor at {descriptor_path}.")

    subprocess.run(
        ["conda", "env", "update", "-p", str(env_path), "-f", str(descriptor_path), "-q"],
        check=True,
    )

def export_shared_environment_descriptor(config: Config) -> None:
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
