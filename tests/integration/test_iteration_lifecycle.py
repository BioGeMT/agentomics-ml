from agentomics.run_logging import logging_helpers
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    get_last_successful_iteration,
    get_next_iteration_index,
    initialize_current_iteration_workspace,
    load_iteration_state,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import save_step_output
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


def test_architecture_table_retains_successful_and_failed_archived_iterations(initialized_run_config, monkeypatch):
    config = initialized_run_config
    prepare_iteration_workspace(config, iteration=0)
    save_step_output(config.current_iteration_dir / "model_architecture", "model_architecture", {
        "architecture": "Random forest", "hyperparameters": "100 trees",
        "categories": {"families": ["random_forest"], "is_ensemble": False},
    })
    save_step_output(config.current_iteration_dir / "model_training", "model_training", {
        "path_to_train_file": "train.py", "path_to_model_file": "training_artifacts/model.pkl",
        "path_to_artifacts_dir": "training_artifacts", "training_summary": "Used a linear fallback",
        "unresolved_issues": None,
        "trained_categories": {"families": ["linear"], "is_ensemble": False},
    })
    update_current_iteration_state(config, ended_at=110.0, status="success")
    archive_current_iteration(config, 0)

    prepare_iteration_workspace(config, iteration=1)
    save_step_output(config.current_iteration_dir / "model_architecture", "model_architecture", {
        "architecture": "Older CNN plan", "hyperparameters": "3 layers",
    })
    update_current_iteration_state(config, ended_at=120.0, status="failed", failed_step="model_training")
    archive_current_iteration(config, 1)

    logged = []
    monkeypatch.setattr(logging_helpers, "is_wandb_active", lambda: True)
    monkeypatch.setattr(logging_helpers.wandb, "log", lambda data, *, step: logged.append((data, step)))

    logging_helpers.log_architecture_table(config, iteration=1)

    payload, step = logged[0]
    table = payload["architectures"]
    assert step == 1
    assert table.columns == ["iteration", "planned", "trained", "ensemble", "architecture", "hyperparameters", "outcome"]
    assert table.data == [
        [0, "random_forest", "linear", False, "Random forest", "100 trees", "success"],
        [1, "", "", None, "Older CNN plan", "3 layers", "failed at model_training"],
    ]
