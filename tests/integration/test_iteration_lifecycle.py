from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    get_last_successful_iteration,
    get_next_iteration_index,
    initialize_current_iteration_workspace,
    load_iteration_state,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.utils.config import Config


def _create_archived_iteration(
    config: Config,
    iteration: int,
    *,
    status: str = "success",
) -> None:
    prepare_iteration_workspace(config, iteration=iteration)
    update_current_iteration_state(config, ended_at=110.0, status=status)
    archive_current_iteration(config, iteration)


def test_archive_moves_current_iteration_to_numbered_directory(initialized_run_config):
    _create_archived_iteration(initialized_run_config, iteration=0)

    assert initialized_run_config.iteration_dir(0).is_dir()
    assert not initialized_run_config.current_iteration_dir.exists()


def test_next_iteration_index_advances_after_each_archive(initialized_run_config):
    assert get_next_iteration_index(initialized_run_config) == 0

    _create_archived_iteration(initialized_run_config, iteration=0)
    assert get_next_iteration_index(initialized_run_config) == 1

    _create_archived_iteration(initialized_run_config, iteration=1)
    assert get_next_iteration_index(initialized_run_config) == 2


def test_last_successful_iteration_skips_failed_iterations(initialized_run_config):
    _create_archived_iteration(initialized_run_config, iteration=0, status="success")
    _create_archived_iteration(initialized_run_config, iteration=1, status="failed")

    assert get_last_successful_iteration(initialized_run_config) == 0


def test_last_successful_iteration_is_none_when_every_iteration_failed(
    initialized_run_config,
):
    _create_archived_iteration(initialized_run_config, iteration=0, status="failed")

    assert get_last_successful_iteration(initialized_run_config) is None


def test_iteration_state_updates_preserve_unchanged_fields(initialized_run_config):
    prepare_iteration_workspace(initialized_run_config, iteration=0)
    original_state = load_iteration_state(initialized_run_config.current_iteration_dir)

    update_current_iteration_state(initialized_run_config, status="running")
    update_current_iteration_state(initialized_run_config, ended_at=112.5)

    state = load_iteration_state(initialized_run_config.current_iteration_dir)
    assert state["started_at"] == original_state["started_at"]
    assert state["ended_at"] == 112.5
    assert state["status"] == "running"


def test_current_iteration_workspace_initialization_is_idempotent(
    initialized_run_config,
):
    initialize_current_iteration_workspace(initialized_run_config)
    initialize_current_iteration_workspace(initialized_run_config)

    assert initialized_run_config.current_iteration_dir.is_dir()
