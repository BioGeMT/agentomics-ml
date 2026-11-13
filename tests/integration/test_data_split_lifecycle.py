import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_split import DataSplitOutput, DataSplitStep
from agentomics.datasets.dataset_preparation import prepare_dataset
from agentomics.run_agent import initialize_run
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    load_iteration_state,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import load_step_output, save_step_output
from agentomics.utils.config import Config
from tests.helpers import create_model_calling_output_tool, create_runtime_step, write_split_folder


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
        config.current_step_dir,
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


@pytest.fixture
def prepared_dataset_run_config(
    config_factory: Callable[..., Config],
    default_dataset_dir: Path,
    tmp_path: Path,
) -> Config:
    """Initialize a run on a prepared dataset that has no validation split."""
    prepared_dir = tmp_path / "prepared" / default_dataset_dir.name
    metadata = prepare_dataset(source_dir=default_dataset_dir, destination_dir=prepared_dir)
    config = config_factory(dataset=prepared_dir.name, datasets_dir=str(prepared_dir.parent))
    initialize_run(config, metadata)
    return config


@pytest.fixture
def prepared_dataset_with_validation_run_config(
    config_factory: Callable[..., Config],
    default_prepared_dataset_dir: Path,
) -> Config:
    """Initialize a run on a prepared dataset that has a validation split."""
    config = config_factory(
        dataset=default_prepared_dataset_dir.name,
        datasets_dir=str(default_prepared_dataset_dir.parent),
    )
    metadata = json.loads((default_prepared_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    initialize_run(config, metadata)
    return config


def _create_symlink_now(link: Path, target: Path) -> None:
    """Create a symlink whose mtime is the precise current time.

    The kernel stamps new files with a coarse clock that can lag datetime.now() by milliseconds,
    so a symlink created right after the step starts could otherwise look older than the step.
    """
    link.symlink_to(target)
    now = time.time()
    os.utime(link, (now, now), follow_symlinks=False)


def _write_generated_split(split_path: Path, dataset_dir: Path, labels_csv: str) -> None:
    """Write a split folder as the split prompt asks: input/ symlinks to dataset files and a new labels.csv."""
    (split_path / "input").mkdir(parents=True)
    _create_symlink_now(split_path / "input" / "data.csv", dataset_dir / "train" / "input" / "data.csv")
    (split_path / "labels.csv").write_text(labels_csv, encoding="utf-8")


def _write_generated_full_split(parent_dir: Path, dataset_dir: Path) -> None:
    _write_generated_split(parent_dir / "train", dataset_dir, "id,numeric_label\ntrain-0,1\n")
    _write_generated_split(parent_dir / "validation", dataset_dir, "id,numeric_label\ntrain-1,0\n")
    _write_generated_split(parent_dir / "mini_train", dataset_dir, "id,numeric_label\ntrain-0,1\n")


def _split_output(train_and_validation_dir: Path, mini_train_dir: Path) -> DataSplitOutput:
    return DataSplitOutput(
        train_path=str(train_and_validation_dir / "train"),
        val_path=str(train_and_validation_dir / "validation"),
        mini_train_path=str(mini_train_dir / "mini_train"),
        splitting_strategy="random split",
    )


def _create_model_writing_files_then_calling_output_tool(
    file_writes_per_attempt: list[Callable[[], None]],
    output: DataSplitOutput,
) -> FunctionModel:
    """Create a local model that, on each attempt, does that attempt's file writes and then returns output.

    The writes happen inside the model call because only files written after the step starts count as new.
    """
    remaining_file_writes = iter(file_writes_per_attempt)

    def respond(_messages, agent_info):
        next(remaining_file_writes)()
        return ModelResponse(
            parts=[ToolCallPart(tool_name=agent_info.output_tools[0].name, args=output.model_dump())]
        )

    return FunctionModel(respond)


def _run_split_step(config: Config, model: FunctionModel, iteration: int = 0) -> DataSplitOutput:
    split_step = create_runtime_step(DataSplitStep, config, model=model)
    split_step.on_iteration_start(iteration)
    asyncio.run(split_step.run())
    return load_step_output(config, DataSplitStep.step_id, config.current_iteration_dir)


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


def test_generated_split_with_input_symlinks_moves_to_versioned_split_dir(
    prepared_dataset_run_config: Config,
):
    config = prepared_dataset_run_config
    prepare_iteration_workspace(config, iteration=0)
    step_dir = config.current_step_dir

    output = _run_split_step(
        config,
        model=_create_model_writing_files_then_calling_output_tool(
            [lambda: _write_generated_full_split(step_dir, config.dataset_dir)],
            _split_output(train_and_validation_dir=step_dir, mini_train_dir=step_dir),
        ),
    )

    split_dir = config.splits_dir / "split_0"
    assert output.train_path == str(split_dir / "train")
    assert output.val_path == str(split_dir / "validation")
    assert output.mini_train_path == str(split_dir / "mini_train")
    assert output.split_version == 0
    assert output.split_changed
    for split_name in ["train", "validation", "mini_train"]:
        data_link = split_dir / split_name / "input" / "data.csv"
        assert data_link.is_symlink()
        assert data_link.readlink().is_absolute()


def test_generated_split_returned_with_unnormalised_paths_is_accepted(
    prepared_dataset_run_config: Config,
):
    config = prepared_dataset_run_config
    prepare_iteration_workspace(config, iteration=0)
    step_dir = config.current_step_dir
    unnormalised_step_dir = step_dir / ".." / step_dir.name

    output = _run_split_step(
        config,
        model=_create_model_writing_files_then_calling_output_tool(
            [lambda: _write_generated_full_split(step_dir, config.dataset_dir)],
            _split_output(train_and_validation_dir=unnormalised_step_dir, mini_train_dir=unnormalised_step_dir),
        ),
    )

    assert output.train_path == str(config.splits_dir / "split_0" / "train")


def test_split_rejected_for_symlink_outside_split_folders_moves_to_versioned_split_dir_once_after_retry(
    prepared_dataset_run_config: Config,
):
    config = prepared_dataset_run_config
    prepare_iteration_workspace(config, iteration=0)
    step_dir = config.current_step_dir
    stray_link = step_dir / "stray_link.csv"

    def first_attempt() -> None:
        _write_generated_full_split(step_dir, config.dataset_dir)
        _create_symlink_now(stray_link, config.dataset_dir / "train" / "input" / "data.csv")

    def second_attempt() -> None:
        stray_link.unlink()

    output = _run_split_step(
        config,
        model=_create_model_writing_files_then_calling_output_tool(
            [first_attempt, second_attempt],
            _split_output(train_and_validation_dir=step_dir, mini_train_dir=step_dir),
        ),
    )

    assert [path.name for path in config.splits_dir.iterdir()] == ["split_0"]
    assert output.train_path == str(config.splits_dir / "split_0" / "train")


def test_mini_train_only_split_links_provided_train_and_validation(
    prepared_dataset_with_validation_run_config: Config,
):
    config = prepared_dataset_with_validation_run_config
    prepare_iteration_workspace(config, iteration=0)
    step_dir = config.current_step_dir

    def write_mini_train() -> None:
        _write_generated_split(
            step_dir / "mini_train", config.dataset_dir, "id,numeric_label\ntrain-0,1\ntrain-1,0\n"
        )

    output = _run_split_step(
        config,
        model=_create_model_writing_files_then_calling_output_tool(
            [write_mini_train],
            _split_output(train_and_validation_dir=config.dataset_dir, mini_train_dir=step_dir),
        ),
    )

    split_dir = config.splits_dir / "split_0"
    assert output.mini_train_path == str(split_dir / "mini_train")
    assert (split_dir / "mini_train" / "labels.csv").is_file()
    for split_name, output_path in [("train", output.train_path), ("validation", output.val_path)]:
        assert output_path == str(split_dir / split_name)
        assert Path(output_path).is_symlink()
        assert Path(output_path).resolve() == (config.dataset_dir / split_name).resolve()
    assert not output.split_changed


def test_reused_split_keeps_version_and_latest_successful_strategy(
    prepared_dataset_run_config: Config,
):
    config = prepared_dataset_run_config
    config.split_allowed_iterations = 2
    existing_split_dir = config.splits_dir / "split_0"
    _write_generated_full_split(existing_split_dir, config.dataset_dir)
    create_archived_iteration_with_split_output(
        create_runtime_step(DataSplitStep, config, model=Mock()),
        iteration=0,
        split_version=0,
        status="success",
        strategy="existing strategy",
    )
    prepare_iteration_workspace(config, iteration=1)

    output = _run_split_step(
        config,
        model=create_model_calling_output_tool(
            _split_output(train_and_validation_dir=existing_split_dir, mini_train_dir=existing_split_dir)
        ),
        iteration=1,
    )

    assert output.train_path == str(existing_split_dir / "train")
    assert output.split_version == 0
    assert output.splitting_strategy == "existing strategy"
    assert not output.split_changed
    assert [path.name for path in config.splits_dir.iterdir()] == ["split_0"]
