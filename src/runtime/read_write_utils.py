from __future__ import annotations

import datetime
import json
import re
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from runtime.conda_utils import export_shared_environment_descriptor
from runtime.filesystem import (
    chown_tree_to_root,
    remove_path,
)
from utils.config import Config

def initialize_run_directories(config: Config) -> None:
    config.markdown_reports_dir.mkdir(parents=True, exist_ok=True)
    config.pdf_reports_dir.mkdir(parents=True, exist_ok=True)
    config.extras_dir.mkdir(parents=True, exist_ok=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.shared_dir.mkdir(parents=True, exist_ok=True)
    config.best_iteration_snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.splits_dir.mkdir(parents=True, exist_ok=True)

def save_config(config: Config) -> None:
    config.config_path.parent.mkdir(parents=True, exist_ok=True)
    config.config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

def load_config(config_path: Path | str, missing_ok: bool = False) -> Config | None:
    config_path = Path(config_path)
    if not config_path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(f"Config not found at {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {config_path}.")
    return Config(**payload)

def load_config_from_run_dir(run_dir: Path | str, missing_ok: bool = False) -> Config | None:
    return load_config(Path(run_dir) / Config.SHARED_DIRNAME / Config.CONFIG_FILENAME, missing_ok=missing_ok)

def load_config_from_run_dir_and_reroot(run_dir: Path) -> Config:
    config = load_config_from_run_dir(run_dir)
    return replace(config, workspace_dir=str(run_dir.parent))

def initialize_current_iteration_workspace(config: Config) -> None:
    config.current_iteration_dir.mkdir(parents=True, exist_ok=True)
    config.current_iteration_runtime_info_dir.mkdir(exist_ok=True)

def archive_current_iteration(config: Config, iteration: int) -> None:
    export_shared_environment_descriptor(config)
    shared_environment_path = config.shared_dir / "environment.yml"
    if shared_environment_path.exists():
        shutil.copy2(shared_environment_path, config.current_iteration_runtime_info_dir / "environment.yml")

    current_iteration_dir = config.current_iteration_dir
    archived_iteration_dir = config.iteration_dir(iteration)
    if archived_iteration_dir.exists():
        remove_path(archived_iteration_dir)
    current_iteration_dir.replace(archived_iteration_dir)
    if config.agent_user:
        chown_tree_to_root(archived_iteration_dir)

def get_archived_iterations(
    config: Config,
    only_successful: bool = False,
) -> list[int]:
    archived_iterations: list[int] = []
    for child in config.run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(Config.ITERATION_DIR_PREFIX):
            continue
        suffix = child.name.removeprefix(Config.ITERATION_DIR_PREFIX)
        if not suffix.isdigit():
            continue
        if only_successful and load_iteration_state(child).get("status") != "success":
            continue
        archived_iterations.append(int(suffix))
    return sorted(archived_iterations)

def get_next_iteration_index(config: Config) -> int:
    archived_iterations = get_archived_iterations(config)
    if not archived_iterations:
        return 0
    return archived_iterations[-1] + 1

def get_last_successful_iteration(config: Config) -> int | None:
    iterations = get_archived_iterations(config, only_successful=True)
    return iterations[-1] if iterations else None

def _get_current_iteration_base_prompt_path(config: Config) -> Path:
    return config.current_iteration_runtime_info_dir / Config.BASE_PROMPT_FILENAME

def _get_current_iteration_system_prompt_path(config: Config) -> Path:
    return config.current_iteration_runtime_info_dir / Config.SYSTEM_PROMPT_FILENAME

def write_current_iteration_base_prompt(config: Config, base_prompt: str) -> None:
    _get_current_iteration_base_prompt_path(config).write_text(base_prompt, encoding="utf-8")

def write_current_iteration_system_prompt(config: Config, system_prompt: str) -> None:
    _get_current_iteration_system_prompt_path(config).write_text(system_prompt, encoding="utf-8")

def load_current_iteration_base_prompt(config: Config) -> str:
    base_prompt_path = _get_current_iteration_base_prompt_path(config)
    if not base_prompt_path.exists():
        raise FileNotFoundError(f"Base prompt file not found at {base_prompt_path}.")
    return base_prompt_path.read_text(encoding="utf-8")

def load_current_iteration_system_prompt(config: Config) -> str:
    system_prompt_path = _get_current_iteration_system_prompt_path(config)
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found at {system_prompt_path}.")
    return system_prompt_path.read_text(encoding="utf-8")

def get_new_rundir_files(config: Config, since_timestamp: datetime.datetime) -> list[str]:
    run_dir = config.current_iteration_dir
    new_files: list[str] = []
    ignored_names = {
        ".conda",
        ".cache",
        "__pycache__",
        ".git",
        ".gitignore",
        Config.RUNTIME_INFO_DIRNAME,
    }
    for element in run_dir.iterdir():
        if element.name in ignored_names:
            continue
        if element.name.startswith(Config.ITERATION_DIR_PREFIX) and element.is_dir():
            continue
        if element.is_file():
            if datetime.datetime.fromtimestamp(element.stat().st_mtime) > since_timestamp:
                new_files.append(element.name)
            continue

        for file_path in element.rglob("*"):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                if datetime.datetime.fromtimestamp(file_path.stat().st_mtime) > since_timestamp:
                    new_files.append(str(file_path.relative_to(run_dir)))
    return new_files

def does_file_contain_string(file_path, search_string) -> bool:
    return _file_matches_quoted_pattern(file_path, rf"(['\"]){re.escape(search_string)}.*?\1")

def does_file_contain_iteration_pattern(file_path) -> bool:
    return _file_matches_quoted_pattern(
        file_path,
        rf"(['\"]){re.escape(Config.ITERATION_DIR_PREFIX)}\d+.*?\1",
    )

def load_iteration_metadata(iteration_dir: Path) -> dict[str, Any]:
    return _load_json_object(iteration_dir / Config.RUNTIME_INFO_DIRNAME / Config.ITERATION_METADATA_FILENAME)

def load_best_iteration_snapshot_iteration(config: Config) -> int | None:
    iteration = load_iteration_metadata(config.best_iteration_snapshot_dir).get("iteration")
    if isinstance(iteration, int):
        return iteration
    return None

def load_iteration_duration(iteration_dir: Path) -> float | None:
    iteration_state = load_iteration_state(iteration_dir)
    started_at = iteration_state.get("started_at")
    ended_at = iteration_state.get("ended_at")
    if started_at is None or ended_at is None:
        return None
    return float(ended_at) - float(started_at)

def load_iteration_state(iteration_dir: Path) -> dict[str, Any]:
    return _load_json_object(iteration_dir / Config.RUNTIME_INFO_DIRNAME / Config.ITERATION_STATE_FILENAME)

def load_current_iteration_index(config: Config) -> int:
    iteration_metadata = load_iteration_metadata(config.current_iteration_dir)
    iteration = iteration_metadata.get("iteration")
    if not isinstance(iteration, int):
        raise KeyError(
            f"Current iteration metadata at {config.current_iteration_runtime_info_dir / Config.ITERATION_METADATA_FILENAME} "
            "is missing integer field 'iteration'."
        )
    return iteration

def initialize_current_iteration_metadata(config: Config, iteration: int) -> None:
    _write_json(
        config.current_iteration_runtime_info_dir / Config.ITERATION_METADATA_FILENAME,
        {"iteration": iteration},
    )

def initialize_current_iteration_state(config: Config, started_at: float) -> None:
    _write_json(
        config.current_iteration_runtime_info_dir / Config.ITERATION_STATE_FILENAME,
        {
            "status": "running",
            "started_at": started_at,
            "ended_at": None,
        },
    )

def update_current_iteration_state(config: Config, **changes: object) -> None:
    iteration_state_path = config.current_iteration_runtime_info_dir / Config.ITERATION_STATE_FILENAME
    iteration_state = _load_json_object(iteration_state_path)
    iteration_state.update(changes)
    _write_json(iteration_state_path, iteration_state)

def load_dataset_metadata(config: Config) -> dict[str, str]:
    return json.loads((config.dataset_dir / "metadata.json").read_text(encoding="utf-8"))

def replace_string_in_tree_files(root_dir: Path, old: str, new: str, skip_dirs: set[str] | None = None) -> None:
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if skip_dirs and any(part in skip_dirs for part in file_path.parts):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        if old in content:
            file_path.write_text(content.replace(old, new), encoding="utf-8")

def _file_matches_quoted_pattern(file_path, pattern: str) -> bool:
    content = Path(file_path).read_text(encoding="utf-8")
    return re.search(pattern, content, re.DOTALL) is not None

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload
