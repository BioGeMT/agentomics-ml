from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from agentomics.utils.config import Config


ENVIRONMENT_ARCHIVE_FILENAME = "environment.tar.gz"

def get_shared_environment_path(config: Config) -> Path:
    return Path("/tmp/agentomics/envs") / f"{config.agent_id}_env"

def get_iteration_environment_descriptor_path(iteration_dir: Path) -> Path:
    return iteration_dir / Config.RUNTIME_INFO_DIRNAME / Config.ENVIRONMENT_DESCRIPTOR_FILENAME


def get_iteration_environment_archive_path(iteration_dir: Path) -> Path:
    return iteration_dir / Config.RUNTIME_INFO_DIRNAME / ENVIRONMENT_ARCHIVE_FILENAME

def restore_iteration_environment(
    iteration_root: Path,
    environment_path: Path,
) -> Path:
    archive_path = get_iteration_environment_archive_path(iteration_root)
    if archive_path.is_file():
        try:
            environment_path.mkdir(parents=True)
            subprocess.run(
                ["tar", "-xf", str(archive_path), "-C", str(environment_path)],
                check=True,
            )
            subprocess.run(
                [str(environment_path / "bin" / "conda-unpack")],
                check=True,
            )
            return environment_path
        except (OSError, subprocess.CalledProcessError):
            shutil.rmtree(environment_path, ignore_errors=True)

    create_environment_from_descriptor(
        get_iteration_environment_descriptor_path(iteration_root),
        env_path=environment_path,
    )
    return environment_path

def export_environment_archive(env_path: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(env_path / "bin" / "conda-pack"),
            "-p", str(env_path),
            "-o", str(archive_path),
            "--force",
        ],
        check=True,
    )

def create_environment_from_descriptor(
    descriptor_path: Path,
    env_path: Path,
) -> None:
    if not descriptor_path.is_file():
        raise FileNotFoundError(f"Missing environment descriptor at {descriptor_path}.")

    subprocess.run(
        [
            "conda",
            "env",
            "create",
            "-p",
            str(env_path),
            "-f",
            str(descriptor_path),
            "-q",
        ],
        check=True,
    )

def update_environment_from_descriptor(descriptor_path: Path, env_path: Path) -> None:
    if not descriptor_path.exists():
        raise FileNotFoundError(f"Missing environment descriptor at {descriptor_path}.")

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
            "--prune",
        ],
        check=True,
    )

def ensure_environment_from_descriptor(descriptor_path: Path, env_path: Path) -> None:
    if env_path.exists():
        update_environment_from_descriptor(descriptor_path, env_path)
        return
    env_path.parent.mkdir(parents=True, exist_ok=True)
    create_environment_from_descriptor(
        descriptor_path,
        env_path=env_path,
    )

def run_python_in_environment(
    environment_path: Path,
    script_path: Path,
    arguments: list[str],
    capture_output: bool = True,
    check: bool = False,
    environment: dict[str, str] | None = None,
):
    command = [
        "conda", "run", "-p", str(environment_path),
        "python", str(script_path),
    ]
    command.extend(arguments)
    return subprocess.run(
        command,
        capture_output=capture_output,
        cwd=script_path.parent,
        check=check,
        env=environment,
    )

def export_shared_environment_descriptor(config: Config) -> None:
    conda_env = get_shared_environment_path(config)
    export_environment_descriptor_to_path(
        env_path=conda_env,
        descriptor_path=config.shared_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME,
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
            lines.append(f"      - {name}=={_strip_local_version(version)}")

    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _collect_environment_packages(
    env_path: Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Query conda-meta and ``uv pip list`` for the authoritative package list.

    Returns (conda_packages, pip_packages) as sorted (name, version) tuples.
    We avoid ``conda env export`` because of long-standing conda bugs.
    """
    conda_records = _read_conda_records(env_path)
    conda_managed: dict[str, tuple[str, str]] = {
        _normalize_package_name(r["name"]): (r["name"], r["version"])
        for r in conda_records
    }

    python_packages = _collect_uv_packages(env_path)
    conda_python_distributions = _collect_conda_python_distributions(env_path, conda_records)

    pip_owned: dict[str, tuple[str, str]] = {}
    conda_replaced_by_pip: set[str] = set()
    for norm, (name, version) in python_packages.items():
        # Some conda packages (e.g. conda-pack) ship dist-info with a placeholder
        # "0.0.0" version that uv reports but PyPI has no matching release. When conda
        # manages the package, trust conda's version rather than misclassifying it as a
        # pip package pinned to 0.0.0 (which makes recreation's pip step fail).
        if version == "0.0.0" and (norm in conda_managed or norm in conda_python_distributions):
            continue
        if norm in conda_managed and version == conda_managed[norm][1]:
            continue
        if norm in conda_python_distributions:
            conda_norm, conda_version = conda_python_distributions[norm]
            if version == conda_version:
                continue
            conda_replaced_by_pip.add(conda_norm)
        pip_owned[norm] = (name, version)

    conda_packages = sorted(
        (
            (name, version)
            for norm, (name, version) in conda_managed.items()
            if norm not in pip_owned and norm not in conda_replaced_by_pip
        ),
        key=lambda x: x[0],
    )
    pip_packages = sorted(pip_owned.values(), key=lambda x: x[0].lower())
    return conda_packages, pip_packages


def _read_conda_records(env_path: Path) -> list[dict]:
    conda_meta_dir = env_path / "conda-meta"
    if not conda_meta_dir.is_dir():
        return []
    records = []
    for meta_file in conda_meta_dir.glob("*.json"):
        record = json.loads(meta_file.read_text(encoding="utf-8"))
        if record.get("name") and record.get("version"):
            records.append(record)
    return records


def _collect_conda_python_distributions(
    env_path: Path, conda_records: list[dict],
) -> dict[str, tuple[str, str]]:
    """Return {python_distribution_name: (conda_package_name, conda_version)}.

    Bridges the gap between conda package names and Python distribution names
    (e.g. conda's ``python-graphviz`` provides Python's ``graphviz``).
    """
    distributions: dict[str, tuple[str, str]] = {}
    for record in conda_records:
        conda_name = record["name"]
        conda_version = record["version"]
        for metadata_path in _iter_python_metadata_paths(env_path, record):
            distribution_name = _read_distribution_name(metadata_path)
            if distribution_name:
                distributions[_normalize_package_name(distribution_name)] = (
                    _normalize_package_name(conda_name),
                    conda_version,
                )
    return distributions


def _iter_python_metadata_paths(env_path: Path, conda_record: dict) -> list[Path]:
    paths = list(conda_record.get("files", []))
    paths.extend(
        path_record.get("_path", "")
        for path_record in conda_record.get("paths_data", {}).get("paths", [])
    )
    return [
        env_path / path
        for path in paths
        if path.endswith((".dist-info/METADATA", ".egg-info/PKG-INFO"))
    ]


def _read_distribution_name(metadata_path: Path) -> str | None:
    if not metadata_path.exists():
        return None
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Name: "):
            return line.removeprefix("Name: ").strip()
    return None


def _collect_uv_packages(env_path: Path) -> dict[str, tuple[str, str]]:
    """Return {normalized_name: (name, version)} from ``uv pip list --format=json``.

    Uses ``uv pip list`` instead of ``uv pip freeze`` because freeze outputs conda
    packages as non-portable ``@ file:///`` direct references.
    """
    python_bin = env_path / "bin" / "python"
    if not python_bin.exists():
        return {}
    uv_bin = _find_uv_binary()
    result = subprocess.run(
        [
            uv_bin, "pip", "list", "--format=json",
            "--python", str(python_bin),
            "--exclude-editable",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    packages: dict[str, tuple[str, str]] = {}
    for entry in json.loads(result.stdout):
        name = entry.get("name", "")
        version = entry.get("version", "")
        if name and version:
            packages[_normalize_package_name(name)] = (name, version)
    return packages


def _find_uv_binary() -> str:
    on_path = shutil.which("uv")
    if on_path:
        return on_path
    next_to_python = Path(sys.executable).parent / "uv"
    if next_to_python.is_file():
        return str(next_to_python)
    raise FileNotFoundError(
        "uv is required for environment export but was not found on PATH "
        "or next to the running Python interpreter. Install it with: pip install uv"
    )


def _strip_local_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
