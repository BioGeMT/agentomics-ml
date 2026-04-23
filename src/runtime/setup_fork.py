from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from runtime.conda_utils import update_environment_from_descriptor
from runtime.git_checkpoints import create_and_checkout_branch_at_checkpoint
from runtime.read_write_utils import load_config_from_run_dir, replace_string_in_tree_files
from utils.config import Config


def fork_run(
    source_workspace_dir: Path,
    target_agent_id: str,
    target_workspace_dir: Path,
    fork_from_step: str | None,
    fork_from_iteration: int | None,
) -> None:
    _validate_fork_args(source_workspace_dir)
    target_workspace_dir.mkdir(parents=True, exist_ok=True)
    if any(target_workspace_dir.iterdir()):
        raise FileExistsError(
            f"Target workspace directory must be empty before fork setup: {target_workspace_dir}"
        )

    source_run_dir = source_workspace_dir / Config.RUN_DIRNAME
    target_run_dir = target_workspace_dir / Config.RUN_DIRNAME
    source_config = load_config_from_run_dir(source_run_dir)

    # 1. Copy entire workspace so the forked run inherits the best iteration snapshot and all outputs
    shutil.copytree(source_workspace_dir, target_workspace_dir, symlinks=False, copy_function=shutil.copy2, dirs_exist_ok=True)

    # 2. Roll the run directory back to the requested checkpoint commit on a new branch
    create_and_checkout_branch_at_checkpoint(
        run_dir=target_run_dir,
        run_id=source_config.agent_id,
        branch_name=f"agentomics-run-{target_agent_id}",
        step_id=fork_from_step,
        iteration=fork_from_iteration,
    )
    # Remove any untracked files left by a step that crashed before its checkpoint commit.
    # -fd removes files and directories; omitting -x preserves gitignored paths (e.g. .conda/).
    subprocess.run(["git", "clean", "-fd"], cwd=target_run_dir, check=True, text=True, capture_output=True)

    # 3. Fix absolute paths in stored step outputs that still point at the source workspace
    replace_string_in_tree_files(target_workspace_dir, str(source_workspace_dir), str(target_workspace_dir), skip_dirs={".conda", ".git"})

    # 4. Rename conda envs: the env directory is named after the agent ID, which changes on fork
    for conda_envs_dir in target_workspace_dir.rglob(".conda/envs"):
        (conda_envs_dir / f"{source_config.agent_id}_env").rename(conda_envs_dir / f"{target_agent_id}_env")

    # 5. Since the conda env is not tracked by git, update the shared env using environment.yml
    update_environment_from_descriptor(
        target_run_dir / "shared" / "environment.yml",
        target_run_dir / "shared" / ".conda" / "envs" / f"{target_agent_id}_env",
    )

def _validate_fork_args(fork_from_run: Path) -> None:
    if not fork_from_run.is_dir():
        raise FileNotFoundError(
            f"--fork-from-run path does not exist: {fork_from_run}\n"
            f"Point it at a run workspace directory (the one containing '{Config.RUN_DIRNAME}/')."
        )
    source_run_dir = fork_from_run / Config.RUN_DIRNAME
    if not source_run_dir.is_dir():
        raise FileNotFoundError(
            f"No '{Config.RUN_DIRNAME}/' subdirectory found under {fork_from_run}."
        )
    if not (source_run_dir / ".git").is_dir():
        raise FileNotFoundError(
            f"Source run at {source_run_dir} has no git repository (.git missing).\n"
            "Only runs with git checkpointing enabled can be forked."
        )

def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a forked run workspace from a source checkpoint.")
    parser.add_argument("--source-workspace", type=Path, required=True,
                        help=f"Path to the source run workspace directory (containing '{Config.RUN_DIRNAME}/').")
    parser.add_argument("--target-workspace", type=Path, required=True,
                        help="Path to the target workspace directory.")
    parser.add_argument("--agent-id", type=str, required=True,
                        help="Agent ID for the new forked run (used as the git branch name).")
    parser.add_argument("--fork-from-step", type=str, default=None,
                        help="Step ID to fork from. Defaults to the latest checkpoint in the source run.")
    parser.add_argument("--fork-from-iteration", type=int, default=None,
                        help="Iteration number to fork from. Defaults to the latest iteration for the selected checkpoint.")
    args = parser.parse_args()

    fork_run(
        source_workspace_dir=args.source_workspace,
        target_agent_id=args.agent_id,
        target_workspace_dir=args.target_workspace,
        fork_from_step=args.fork_from_step,
        fork_from_iteration=args.fork_from_iteration,
    )

if __name__ == "__main__":
    _main()
