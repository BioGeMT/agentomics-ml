from pathlib import Path

import pytest

from agentomics.runtime.git_checkpoints import (
    commit_iteration_end,
    commit_step_checkpoint,
    create_and_checkout_branch_at_checkpoint,
)
from agentomics.utils.config import Config
from tests.helpers import run_git_cli_command


def test_checks_out_requested_step_and_iteration(initialized_run_config: Config):
    repo_dir = Path(initialized_run_config.workspace_dir)
    commit_step_checkpoint(
        initialized_run_config,
        iteration=0,
        step_id="model_training",
    )
    expected_hash = run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip()
    commit_step_checkpoint(
        initialized_run_config,
        iteration=1,
        step_id="model_training",
    )

    create_and_checkout_branch_at_checkpoint(
        repo_dir,
        initialized_run_config.agent_id,
        "fork-branch",
        step_id="model_training",
        iteration=0,
    )

    assert run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip() == expected_hash
    assert run_git_cli_command(repo_dir, "branch", "--show-current").stdout.strip() == "fork-branch"


def test_checks_out_latest_checkpoint_for_requested_step(initialized_run_config: Config):
    repo_dir = Path(initialized_run_config.workspace_dir)
    commit_step_checkpoint(
        initialized_run_config,
        iteration=0,
        step_id="model_training",
    )
    commit_step_checkpoint(
        initialized_run_config,
        iteration=1,
        step_id="model_training",
    )
    expected_hash = run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip()
    commit_step_checkpoint(
        initialized_run_config,
        iteration=1,
        step_id="model_inference",
    )

    create_and_checkout_branch_at_checkpoint(
        repo_dir,
        initialized_run_config.agent_id,
        "fork-branch",
        step_id="model_training",
    )

    assert run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip() == expected_hash
    assert run_git_cli_command(repo_dir, "branch", "--show-current").stdout.strip() == "fork-branch"


def test_checks_out_iteration_end_checkpoint(initialized_run_config: Config):
    repo_dir = Path(initialized_run_config.workspace_dir)
    commit_step_checkpoint(
        initialized_run_config,
        iteration=0,
        step_id="model_training",
    )
    commit_iteration_end(initialized_run_config, iteration=0)
    expected_hash = run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip()
    commit_step_checkpoint(
        initialized_run_config,
        iteration=1,
        step_id="model_training",
    )

    create_and_checkout_branch_at_checkpoint(
        repo_dir,
        initialized_run_config.agent_id,
        "fork-branch",
        iteration=0,
    )

    assert run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip() == expected_hash
    assert run_git_cli_command(repo_dir, "branch", "--show-current").stdout.strip() == "fork-branch"


@pytest.mark.parametrize(
    ("step_id", "iteration"),
    [
        pytest.param("nonexistent_step", None, id="missing-step"),
        pytest.param("model_training", 5, id="missing-iteration"),
    ],
)
def test_missing_checkpoint_leaves_repository_unchanged(
    initialized_run_config: Config,
    step_id: str,
    iteration: int | None,
):
    repo_dir = Path(initialized_run_config.workspace_dir)
    commit_step_checkpoint(
        initialized_run_config,
        iteration=0,
        step_id="model_training",
    )
    original_head = run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(ValueError):
        create_and_checkout_branch_at_checkpoint(
            repo_dir,
            initialized_run_config.agent_id,
            "fork-branch",
            step_id=step_id,
            iteration=iteration,
        )

    assert run_git_cli_command(repo_dir, "rev-parse", "HEAD").stdout.strip() == original_head
    assert not run_git_cli_command(
        repo_dir,
        "branch",
        "--list",
        "fork-branch",
    ).stdout.strip()
