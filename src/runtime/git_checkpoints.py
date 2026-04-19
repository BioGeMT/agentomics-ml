from __future__ import annotations

import subprocess

from utils.config import Config


def initialize_repo_if_needed(config: Config) -> None:
    run_dir = config.run_dir
    git_dir = run_dir / ".git"
    if not git_dir.exists():
        _git(config, ["init"])

    _configure_repo(config)
    _write_gitignore(config)

def commit_run_start_if_needed(config: Config) -> None:
    _commit_all(config, build_run_start_commit_message(config.agent_id))

def commit_step_checkpoint(config: Config, iteration: int, step_id: str) -> None:
    _commit_all(config, build_step_commit_message(config.agent_id, iteration, step_id))

def commit_iteration_end(config: Config, iteration: int) -> None:
    _commit_all(config, build_iteration_end_commit_message(config.agent_id, iteration))

def build_step_commit_message(run_id: str, iteration: int, step_id: str) -> str:
    return f"agentomics: run={run_id} iteration={iteration:04d} step={step_id}"

def build_iteration_end_commit_message(run_id: str, iteration: int) -> str:
    return f"agentomics: run={run_id} iteration={iteration:04d} end"

def build_run_start_commit_message(run_id: str) -> str:
    return f"agentomics: run={run_id} start"

def _configure_repo(config: Config) -> None:
    _git(config, ["config", "user.name", "Agentomics Runtime"], check=False)
    _git(config, ["config", "user.email", "agentomics@local"], check=False)

def _write_gitignore(config: Config) -> None:
    gitignore_path = config.run_dir / ".gitignore"
    gitignore_lines = [
        "shared/.conda/",
        "__pycache__/",
        ".cache/",
        "*.pyc",
    ]
    gitignore_path.write_text("\n".join(gitignore_lines) + "\n", encoding="utf-8")

def _commit_all(config: Config, message: str) -> None:
    _git(config, ["add", "-A"])
    status = _git(config, ["status", "--porcelain"], capture_output=True)
    if not status.stdout.strip():
        return
    _git(config, ["commit", "-m", message])

def _git(
    config: Config,
    args: list[str],
    capture_output: bool = False,
    check: bool = True,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    should_capture_output = capture_output or not verbose
    return subprocess.run(
        ["git", *args],
        cwd=config.run_dir,
        check=check,
        text=True,
        stdout=subprocess.PIPE if should_capture_output else None,
        stderr=subprocess.PIPE if should_capture_output else None,
    )
