import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _get_env_path(*names: str) -> Optional[Path]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return _resolve_path(value)
    return None


def find_repo_root(start: str | Path | None = None) -> Optional[Path]:
    current = _resolve_path(start or Path.cwd())
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "agentomics").exists():
            return candidate
    return None


def get_default_base_dir(cwd: str | Path | None = None) -> Path:
    current = _resolve_path(cwd or Path.cwd())
    return find_repo_root(current) or current


@dataclass(frozen=True)
class AgentomicsPaths:
    base_dir: Path
    workspace_dir: Path
    datasets_dir: Path
    prepared_datasets_dir: Path
    prepared_test_sets_dir: Path
    agent_datasets_dir: Path


def resolve_agentomics_paths(
    cwd: str | Path | None = None,
    *,
    workspace_dir: str | Path | None = None,
    datasets_dir: str | Path | None = None,
    prepared_datasets_dir: str | Path | None = None,
    prepared_test_sets_dir: str | Path | None = None,
    agent_datasets_dir: str | Path | None = None,
) -> AgentomicsPaths:
    base_dir = get_default_base_dir(cwd)

    resolved_workspace_dir = (
        _resolve_path(workspace_dir)
        if workspace_dir is not None
        else _get_env_path("AGENTOMICS_WORKSPACE_DIR") or (base_dir / "workspace").resolve()
    )
    resolved_datasets_dir = (
        _resolve_path(datasets_dir)
        if datasets_dir is not None
        else _get_env_path("DATASETS_DIR") or (base_dir / "datasets").resolve()
    )
    resolved_prepared_datasets_dir = (
        _resolve_path(prepared_datasets_dir)
        if prepared_datasets_dir is not None
        else _get_env_path("PREPARED_DATASETS_DIR") or (base_dir / "prepared_datasets").resolve()
    )
    resolved_prepared_test_sets_dir = (
        _resolve_path(prepared_test_sets_dir)
        if prepared_test_sets_dir is not None
        else _get_env_path("PREPARED_TEST_SETS_DIR", "PREPARED_TESTS_DIR")
        or (base_dir / "prepared_test_sets").resolve()
    )
    resolved_agent_datasets_dir = (
        _resolve_path(agent_datasets_dir)
        if agent_datasets_dir is not None
        else _get_env_path("AGENTOMICS_AGENT_DATASETS_DIR") or (resolved_workspace_dir / "datasets").resolve()
    )

    return AgentomicsPaths(
        base_dir=base_dir,
        workspace_dir=resolved_workspace_dir,
        datasets_dir=resolved_datasets_dir,
        prepared_datasets_dir=resolved_prepared_datasets_dir,
        prepared_test_sets_dir=resolved_prepared_test_sets_dir,
        agent_datasets_dir=resolved_agent_datasets_dir,
    )
