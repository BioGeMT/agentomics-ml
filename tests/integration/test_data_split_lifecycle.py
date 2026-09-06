import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from agentomics.agents.steps.data_split import DataSplitOutput, DataSplitStep
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    load_iteration_state,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import save_step_output
from agentomics.utils.config import Config
from tests.helpers import create_runtime_step, write_split_folder


def _create_versioned_split(config: Config, version: int) -> Path:
    split_dir = config.splits_dir / f"split_{version}"
    write_split_folder(split_dir / "train", row_id="train-row")
    write_split_folder(split_dir / "validation", row_id="validation-row")
    write_split_folder(split_dir / "mini_train", row_id="train-row")
    return split_dir


def create_archived_iteration_with_split_output(
    step: DataSplitStep,
    *,
    iteration: int,
    split_version: int,
    status: str,
    strategy: str | None = None,
) -> None:
    config = step.config
    prepare_iteration_workspace(config, iteration=iteration)
    update_current_iteration_state(config, status=status, ended_at=time.time())
    split_dir = config.splits_dir / f"split_{split_version}"
    save_step_output(
        config,
        step.step_id,
        DataSplitOutput(
            train_path=str(split_dir / "train"),
            val_path=str(split_dir / "validation"),
            mini_train_path=str(split_dir / "mini_train"),
            splitting_strategy=(
                strategy or f"strategy for split {split_version}"
            ),
            split_changed=False,
            split_version=split_version,
        ),
    )
    step.archive_step_folder()
    archive_current_iteration(config, iteration)


@pytest.mark.parametrize(
    (
        "has_reusable_split",
        "has_provided_validation",
        "iteration",
        "split_budget",
        "expected_full_split",
        "expected_only_mini_split",
        "expected_simulated",
    ),
    [
        pytest.param(False, False, 3, 0, True, False, False, id="initial-split"),
        pytest.param(True, False, 0, 1, True, False, False, id="budget-remains"),
        pytest.param(True, False, 1, 1, False, False, True, id="budget-exhausted"),
        pytest.param(False, True, 0, 1, False, True, False, id="mini-train-only"),
        pytest.param(True, True, 0, 1, False, False, True, id="provided-reusable"),
    ],
)
def test_iteration_start_selects_available_split_mode(
    initialized_run_config: Config,
    prepared_iteration,
    has_reusable_split: bool,
    has_provided_validation: bool,
    iteration: int,
    split_budget: int,
    expected_full_split: bool,
    expected_only_mini_split: bool,
    expected_simulated: bool,
):
    split_step = create_runtime_step(
        DataSplitStep,
        initialized_run_config,
        model=Mock(),
    )
    config = split_step.config
    config.split_allowed_iterations = split_budget
    if has_reusable_split:
        _create_versioned_split(config, version=0)
    if has_provided_validation:
        write_split_folder(
            config.dataset_dir / "validation",
            row_id="validation-row",
        )

    split_step.on_iteration_start(iteration)

    state = load_iteration_state(config.current_iteration_dir)
    assert state["full_split_allowed_at_start"] is expected_full_split
    assert state["only_mini_split_allowed_at_start"] is expected_only_mini_split
    assert split_step.should_be_simulated() is expected_simulated


@pytest.mark.parametrize(
    ("deadline_offset", "iteration", "split_budget", "expected_full_split"),
    [
        pytest.param(3600, 5, 0, True, id="future-deadline"),
        pytest.param(-3600, 0, 10, False, id="past-deadline"),
    ],
)
def test_split_deadline_overrides_iteration_budget(
    initialized_run_config: Config,
    prepared_iteration,
    deadline_offset: int,
    iteration: int,
    split_budget: int,
    expected_full_split: bool,
):
    split_step = create_runtime_step(
        DataSplitStep,
        initialized_run_config,
        model=Mock(),
    )
    config = split_step.config
    _create_versioned_split(config, version=0)
    config.split_allowed_iterations = split_budget
    config.split_time_deadline = int(time.time()) + deadline_offset

    split_step.on_iteration_start(iteration)

    state = load_iteration_state(config.current_iteration_dir)
    assert state["full_split_allowed_at_start"] is expected_full_split
    assert split_step.should_be_simulated() is not expected_full_split


def test_simulated_output_reuses_latest_split_and_latest_successful_strategy(
    initialized_run_config: Config,
):
    split_step = create_runtime_step(
        DataSplitStep,
        initialized_run_config,
        model=Mock(),
    )
    config = split_step.config
    _create_versioned_split(config, version=0)
    latest_split_dir = _create_versioned_split(config, version=1)
    create_archived_iteration_with_split_output(
        split_step,
        iteration=0,
        split_version=0,
        status="success",
    )
    create_archived_iteration_with_split_output(
        split_step,
        iteration=1,
        split_version=1,
        status="success",
    )
    create_archived_iteration_with_split_output(
        split_step,
        iteration=2,
        split_version=1,
        status="failed",
        strategy="failed iteration strategy",
    )

    output = split_step.build_simulated_output()

    assert Path(output.train_path) == latest_split_dir / "train"
    assert Path(output.val_path) == latest_split_dir / "validation"
    assert Path(output.mini_train_path) == latest_split_dir / "mini_train"
    assert output.splitting_strategy == "strategy for split 1"
    assert output.split_version == 1
    assert not output.split_changed


def test_simulated_output_creates_symlinks_to_provided_dataset_splits(
    initialized_run_config: Config,
):
    split_step = create_runtime_step(
        DataSplitStep,
        initialized_run_config,
        model=Mock(),
    )
    config = split_step.config
    write_split_folder(
        config.dataset_dir / "validation",
        row_id="validation-row",
    )
    write_split_folder(config.dataset_dir / "mini_train", row_id="train-row")

    output = split_step.build_simulated_output()

    split_paths = {
        "train": Path(output.train_path),
        "validation": Path(output.val_path),
        "mini_train": Path(output.mini_train_path),
    }
    for split_name, split_path in split_paths.items():
        assert split_path.is_symlink()
        assert split_path.resolve() == (config.dataset_dir / split_name).resolve()
    assert output.split_version == 0
    assert not output.split_changed
    assert output.splitting_strategy == ""


@pytest.mark.parametrize("missing_split", ["validation", "mini_train"])
def test_simulated_output_requires_complete_provided_dataset_splits(
    initialized_run_config: Config,
    missing_split: str,
):
    split_step = create_runtime_step(
        DataSplitStep,
        initialized_run_config,
        model=Mock(),
    )
    provided_split = "mini_train" if missing_split == "validation" else "validation"
    write_split_folder(split_step.config.dataset_dir / provided_split)

    with pytest.raises(AssertionError, match="ensure split folders are available"):
        split_step.build_simulated_output()
