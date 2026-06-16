from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from runtime.filesystem import remove_path
from utils.config import Config


def get_shared_conda_root(config: Config) -> Path:
    return config.shared_dir / ".conda"


def get_shared_environment_path(config: Config) -> Path:
    return get_shared_conda_root(config) / "envs" / f"{config.agent_id}_env"


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
        ["conda", "env", "update", "-p", str(env_path), "-f", str(descriptor_path), "-q", "--prune"],
        check=True,
    )

def ensure_environment_from_descriptor(descriptor_path: Path, env_path: Path) -> None:
    if env_path.exists():
        update_environment_from_descriptor(descriptor_path, env_path)
        return
    create_environment_from_descriptor(descriptor_path, env_path)

def export_shared_environment_descriptor(config: Config) -> None:
    conda_env = get_shared_environment_path(config)
    export_environment_descriptor_to_path(
        env_path=conda_env,
        descriptor_path=config.shared_dir / "environment.yml",
    )

def export_environment_descriptor_to_path(env_path: Path, descriptor_path: Path) -> None:
    if not env_path.exists():
        return

    conda_packages, pip_packages = _collect_environment_packages(env_path)

    lines = ["channels:", "  - conda-forge", "dependencies:"]
    for name, version in conda_packages:
        lines.append(f"  - {name}={version}")
    if pip_packages:
        lines.append("  - pip")
        lines.append("  - pip:")
        for name, version in pip_packages:
            lines.append(f"      - {name}=={version.split('+', maxsplit=1)[0]}")

    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _collect_environment_packages(
    env_path: Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Query conda-meta and ``pip inspect`` for the authoritative package list.

    Returns (conda_packages, pip_packages) as sorted (name, version) tuples.

    We avoid ``conda env export`` because of long-standing conda bugs
    """
    conda_managed: dict[str, tuple[str, str, Path]] = {}
    conda_meta_dir = env_path / "conda-meta"
    if conda_meta_dir.is_dir():
        for meta_file in conda_meta_dir.glob("*.json"):
            record = json.loads(meta_file.read_text(encoding="utf-8"))
            name = record.get("name", "")
            version = record.get("version", "")
            if name and version:
                conda_managed[_normalize_package_name(name)] = (name, version, meta_file)

    pip_candidates = _collect_pip_inspect_packages(env_path)

    pip_owned: dict[str, tuple[str, str]] = {}
    for norm, (name, version, dist_info_path) in pip_candidates.items():
        if norm in conda_managed:
            _, conda_ver, conda_meta_file = conda_managed[norm]
            if conda_ver == version:
                continue
            # Both claim the package at different versions — the newer metadata wins.
            if conda_meta_file.stat().st_mtime >= dist_info_path.stat().st_mtime:
                continue
        pip_owned[norm] = (name, version)

    conda_packages = sorted(
        ((n, v) for norm, (n, v, _) in conda_managed.items() if norm not in pip_owned),
        key=lambda x: x[0],
    )
    pip_packages = sorted(pip_owned.values(), key=lambda x: x[0].lower())
    return conda_packages, pip_packages


def _collect_pip_inspect_packages(env_path: Path) -> dict[str, tuple[str, str, Path]]:
    """Return {normalized_name: (name, version, dist_info_path)} for pip-installed packages."""
    pip_bin = env_path / "bin" / "pip"
    if not pip_bin.exists():
        return {}
    result = subprocess.run(
        [str(pip_bin), "inspect"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    packages: dict[str, tuple[str, str, Path]] = {}
    for dist in json.loads(result.stdout).get("installed", []):
        if dist.get("installer", "") != "pip":
            continue
        name = dist.get("metadata", {}).get("name", "")
        version = dist.get("metadata", {}).get("version", "")
        location = dist.get("metadata_location", "")
        if name and version and location:
            packages[_normalize_package_name(name)] = (name, version, Path(location))
    return packages


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
