from __future__ import annotations

import subprocess
from pathlib import Path

from agentomics.utils.config import Config


def initialize_repo_if_needed(config: Config) -> None:
    workspace_dir = Path(config.workspace_dir)
    if not (workspace_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=workspace_dir, check=True, text=True, capture_output=True)
    _configure_repo(config)
    _write_gitignore(config)

def commit_step_checkpoint(config: Config, iteration: int, step_id: str) -> None:
    _commit_all(Path(config.workspace_dir), _step_message(config.agent_id, iteration, step_id))

def commit_iteration_end(config: Config, iteration: int) -> None:
    _commit_all(Path(config.workspace_dir), _iteration_end_message(config.agent_id, iteration))

def build_step_commit_message(run_id: str, iteration: int, step_id: str) -> str:
    return _step_message(run_id, iteration, step_id)

def build_iteration_end_commit_message(run_id: str, iteration: int) -> str:
    return _iteration_end_message(run_id, iteration)

def create_and_checkout_branch_at_checkpoint(
    workspace_dir: Path,
    run_id: str,
    branch_name: str,
    step_id: str | None = None,
    iteration: int | None = None,
) -> None:
    commit_hash = _find_checkpoint_commit(workspace_dir, run_id, step_id, iteration)
    # The copied workspace may have files written after the last git commit (e.g. wandb
    # binary logs). Reset to HEAD so the working tree is clean before switching branches.
    subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=workspace_dir, check=True, text=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", branch_name, commit_hash],
        cwd=workspace_dir, check=True, text=True, capture_output=True,
    )

def _find_checkpoint_commit(
    workspace_dir: Path,
    run_id: str,
    step_id: str | None,
    iteration: int | None,
) -> str:
    result = subprocess.run(
        ["git", "log", "--format=%H %s", "--all"],
        cwd=workspace_dir, check=True, text=True, capture_output=True,
    )
    # git log is newest-first so the first match is always the latest
    # Checkpoint commits have the format:
    #   agentomics/{run_id}/{iteration:04d}/{step_id}
    #   agentomics/{run_id}/{iteration:04d}/end
    for line in result.stdout.splitlines():
        commit_hash, _, message = line.partition(" ")
        message = message.strip()

        if not message.startswith(f"agentomics/{run_id}/"):
            continue
        after_run_id = message.removeprefix(f"agentomics/{run_id}/")
        iter_part, slash, step_part = after_run_id.partition("/")
        if not slash or not iter_part.isdigit():
            continue

        commit_iter, commit_step = int(iter_part), step_part
        # If iteration or step are not specified, we take the latest
        iter_matches = iteration is None or commit_iter == iteration
        step_matches = step_id is None or commit_step == step_id
        if iter_matches and step_matches:
            return commit_hash

    raise ValueError(
        f"No checkpoint found for run={run_id!r}"
        + (f", step={step_id!r}" if step_id else "")
        + (f", iteration={iteration}" if iteration is not None else "")
        + f".\nCommits in source run:\n{result.stdout}"
    )

def _step_message(run_id: str, iteration: int, step_id: str) -> str:
    return f"agentomics/{run_id}/{iteration:04d}/{step_id}"

def _iteration_end_message(run_id: str, iteration: int) -> str:
    return f"agentomics/{run_id}/{iteration:04d}/end"

def _configure_repo(config: Config) -> None:
    subprocess.run(["git", "config", "user.name", "Agentomics Runtime"], cwd=config.workspace_dir, check=False, text=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agentomics@local"], cwd=config.workspace_dir, check=False, text=True, capture_output=True)

def _write_gitignore(config: Config) -> None:
    gitignore_lines = [
        "run/shared/.conda/",
        f"{Config.BEST_ITERATION_SNAPSHOT_DIRNAME}/.conda/",
        "__pycache__/",
        ".cache/",
        "*.pyc",
    ]
    (Path(config.workspace_dir) / ".gitignore").write_text("\n".join(gitignore_lines) + "\n", encoding="utf-8")

def _commit_all(workspace_dir: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=workspace_dir, check=True, text=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", message], cwd=workspace_dir, check=True, text=True, capture_output=True)
