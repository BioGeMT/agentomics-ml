import math
from collections import Counter
from types import SimpleNamespace

from agentomics.run_logging import logging_helpers
from agentomics.run_logging.logging_helpers import log_main_metric, log_run_metrics
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


def test_run_metrics_sum_tokens_over_all_steps_of_the_iteration():
    token_usage = {
        "data_exploration": Counter(input_tokens=300, output_tokens=40, tool_calls=3),
        "model_training": Counter(input_tokens=200, output_tokens=10, tool_calls=2),
    }

    values = log_run_metrics(
        iteration=0,
        iteration_state={"started_at": 100.0, "ended_at": 160.5, "status": "success"},
        token_usage=token_usage,
    )

    assert values == {"input_tokens": 500, "output_tokens": 50, "tool_calls": 5, "duration": 60.5, "failed": 0}


def test_run_metrics_flag_a_failed_iteration():
    values = log_run_metrics(
        iteration=0,
        iteration_state={"started_at": 100.0, "ended_at": 101.0, "status": "failed"},
        token_usage={},
    )

    assert values["failed"] == 1


def test_main_metric_is_the_configured_validation_metric():
    metrics = {"validation/AUROC": 0.91, "validation/ACC": 0.8, "train/AUROC": 0.99}

    assert log_main_metric(iteration=0, metric_name="AUROC", metrics=metrics) == 0.91


def test_main_metric_is_missing_when_validation_did_not_report_it():
    assert math.isnan(log_main_metric(iteration=0, metric_name="AUROC", metrics={}))


def test_run_metrics_are_logged_to_wandb_under_one_prefix_with_running_totals(monkeypatch):
    logged = {}

    run = SimpleNamespace(summary={})

    def fake_log(payload, step):
        logged[step] = payload
        run.summary.update(payload)

    monkeypatch.setattr(logging_helpers.wandb, "run", run)
    monkeypatch.setattr(logging_helpers.wandb, "log", fake_log)

    for iteration, status in enumerate(["success", "failed"]):
        log_run_metrics(
            iteration=iteration,
            iteration_state={"started_at": 0.0, "ended_at": 10.0, "status": status},
            token_usage={"model_training": Counter(input_tokens=100, output_tokens=5, tool_calls=4)},
        )

    assert logged[1] == {
        "run_metrics/input_tokens": 100,
        "run_metrics/output_tokens": 5,
        "run_metrics/tool_calls": 4,
        "run_metrics/duration": 10.0,
        "run_metrics/failed": 1,
        "run_metrics/input_tokens_total": 200,
        "run_metrics/output_tokens_total": 10,
        "run_metrics/tool_calls_total": 8,
        "run_metrics/duration_total": 20.0,
        "run_metrics/failed_total": 1,
    }
