import time
from unittest.mock import Mock

import pytest

from agentomics.agents.steps.data_split import DataSplitOutput, DataSplitStep
from agentomics.agents.steps.validation_evaluation import (
    ValidationEvaluationOutput,
    ValidationEvaluationStep,
)
from agentomics.runtime.best_iteration_snapshot import update_best_iteration_snapshot
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import save_step_output
from agentomics.utils.config import Config
from tests.helpers import create_runtime_step, write_split_folder


def _write_split_version(
    config: Config,
    split_version: int,
) -> None:
    split_dir = config.splits_dir / f"split_{split_version}"
    for split_name in ("train", "validation", "mini_train"):
        write_split_folder(split_dir / split_name)
    save_step_output(
        config,
        DataSplitStep.step_id,
        DataSplitOutput(
            train_path=str(split_dir / "train"),
            val_path=str(split_dir / "validation"),
            mini_train_path=str(split_dir / "mini_train"),
            splitting_strategy="test split",
            split_changed=False,
            split_version=split_version,
        ),
    )
    create_runtime_step(DataSplitStep, config, model=Mock()).archive_step_folder()


def _write_existing_best(
    config: Config,
    metrics: dict[str, float],
    *,
    split_version: int = 0,
) -> None:
    best_iteration = 0
    prepare_iteration_workspace(config, iteration=best_iteration)
    _write_split_version(config, split_version)
    save_step_output(
        config,
        ValidationEvaluationStep.step_id,
        ValidationEvaluationOutput(
            metrics=metrics,
            is_new_best=True,
            status="success",
        ),
    )
    create_runtime_step(
        ValidationEvaluationStep,
        config,
        model=Mock(),
    ).archive_step_folder()
    update_current_iteration_state(config, status="success", ended_at=time.time())
    archive_current_iteration(config, best_iteration)
    update_best_iteration_snapshot(config, iteration=best_iteration)


@pytest.mark.parametrize(
    ("metric_name", "best_value", "new_value", "expected"),
    [
        pytest.param("ACC", 0.8, 0.9, True, id="higher-score-improves-ACC"),
        pytest.param("ACC", 0.8, 0.7, False, id="lower-score-worsens-ACC"),
        pytest.param("ACC", 0.8, 0.8, False, id="equal-ACC-is-not-better"),
        pytest.param("LOG_LOSS", 0.8, 0.7, True, id="lower-score-improves-LOG_LOSS"),
        pytest.param("LOG_LOSS", 0.8, 0.9, False, id="higher-score-worsens-LOG_LOSS"),
        pytest.param("LOG_LOSS", 0.8, 0.8, False, id="equal-LOG_LOSS-is-not-better"),
    ],
)
def test_same_split_uses_configured_metric_direction(
    initialized_run_with_lightweight_environment: Config,
    metric_name: str,
    best_value: float,
    new_value: float,
    expected: bool,
):
    config = initialized_run_with_lightweight_environment
    config.val_metric = metric_name
    validation_metric = f"validation/{metric_name}"
    _write_existing_best(config, {validation_metric: best_value})
    prepare_iteration_workspace(config, iteration=1)
    _write_split_version(config, split_version=0)

    is_new_best = ValidationEvaluationStep.is_new_best_iteration(
        config,
        {validation_metric: new_value},
    )

    assert is_new_best is expected


def test_different_split_version_is_new_best_without_comparing_scores(
    initialized_run_with_lightweight_environment: Config,
):
    config = initialized_run_with_lightweight_environment
    config.val_metric = "ACC"
    _write_existing_best(
        config,
        {"validation/ACC": 0.99},
        split_version=0,
    )
    prepare_iteration_workspace(config, iteration=1)
    _write_split_version(config, split_version=1)

    is_new_best = ValidationEvaluationStep.is_new_best_iteration(
        config,
        {"validation/ACC": 0.5},
    )

    assert is_new_best


def test_first_valid_validation_score_is_new_best(config_factory):
    config = config_factory(val_metric="ACC")

    is_new_best = ValidationEvaluationStep.is_new_best_iteration(
        config,
        {"validation/ACC": 0.1},
    )

    assert is_new_best


def test_train_score_without_validation_score_is_not_new_best(config_factory):
    config = config_factory(val_metric="ACC")

    is_new_best = ValidationEvaluationStep.is_new_best_iteration(
        config,
        {"train/ACC": 1.0},
    )

    assert not is_new_best
