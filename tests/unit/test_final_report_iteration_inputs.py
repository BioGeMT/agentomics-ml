import json
import os
import time
from unittest.mock import Mock

from agentomics.agents.steps.data_split import DataSplitOutput, DataSplitStep
from agentomics.agents.steps.validation_evaluation import (
    ValidationEvaluationOutput,
    ValidationEvaluationStep,
)
from agentomics.runtime.best_iteration_snapshot import update_best_iteration_snapshot
from agentomics.runtime.generate_final_reports import gather_iteration_inputs
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import save_step_output
from agentomics.utils.config import Config
from tests.helpers import create_runtime_step, write_split_folder


def _save_data_split_output(config: Config, split_version: int, strategy: str) -> None:
    split_dir = config.splits_dir / f"split_{split_version}"
    save_step_output(
        config,
        DataSplitStep.step_id,
        DataSplitOutput(
            train_path=str(split_dir / "train"),
            val_path=str(split_dir / "validation"),
            mini_train_path=str(split_dir / "mini_train"),
            splitting_strategy=strategy,
            split_changed=bool(split_version),
            split_version=split_version,
        ),
    )
    create_runtime_step(DataSplitStep, config, model=Mock()).archive_step_folder()


def test_iteration_inputs_use_labels_from_its_split_version(
    initialized_run_config: Config,
):
    config = initialized_run_config
    write_split_folder(config.splits_dir / "split_0" / "train", row_id="old-train")
    write_split_folder(
        config.splits_dir / "split_0" / "validation",
        row_id="old-validation",
    )
    write_split_folder(config.splits_dir / "split_1" / "train", row_id="new-train")
    write_split_folder(
        config.splits_dir / "split_1" / "validation",
        row_id="new-validation",
    )
    write_split_folder(config.splits_dir / "split_1" / "mini_train")
    prepare_iteration_workspace(config, iteration=0)
    _save_data_split_output(config, split_version=1, strategy="new split")
    update_current_iteration_state(config, status="success", ended_at=time.time())
    archive_current_iteration(config, iteration=0)

    inputs = gather_iteration_inputs(config=config, iteration=0)

    labels_by_split = {
        split.split_name: split.labeled_csv for split in inputs.splits
    }
    assert labels_by_split["train"] == (
        config.splits_dir / "split_1" / "train" / "labels.csv"
    )
    assert labels_by_split["validation"] == (
        config.splits_dir / "split_1" / "validation" / "labels.csv"
    )


def test_iteration_inputs_include_readable_labels_from_symlinked_splits(
    initialized_run_config: Config,
):
    config = initialized_run_config
    write_split_folder(config.dataset_dir / "train", row_id="ds-train")
    write_split_folder(
        config.dataset_dir / "validation",
        row_id="ds-validation",
    )
    split_dir = config.splits_dir / "split_0"
    split_dir.mkdir(parents=True)
    os.symlink(config.dataset_dir / "train", split_dir / "train")
    os.symlink(config.dataset_dir / "validation", split_dir / "validation")
    write_split_folder(split_dir / "mini_train", row_id="ds-train")
    prepare_iteration_workspace(config, iteration=0)
    _save_data_split_output(config, split_version=0, strategy="")
    update_current_iteration_state(config, status="success", ended_at=time.time())
    archive_current_iteration(config, iteration=0)

    inputs = gather_iteration_inputs(config=config, iteration=0)

    labels_by_split = {
        split.split_name: split.labeled_csv for split in inputs.splits
    }
    assert "ds-train" in labels_by_split["train"].read_text(encoding="utf-8")
    assert "ds-validation" in labels_by_split["validation"].read_text(
        encoding="utf-8"
    )


def test_iteration_inputs_include_test_artifacts_from_best_snapshot(
    initialized_run_with_lightweight_environment: Config,
):
    config = initialized_run_with_lightweight_environment
    config.conda_export_mode = "yaml"
    split_dir = config.splits_dir / "split_0"
    for split_name in ("train", "validation", "mini_train"):
        write_split_folder(split_dir / split_name)
    prepare_iteration_workspace(config, iteration=0)
    _save_data_split_output(config, split_version=0, strategy="")
    save_step_output(
        config,
        ValidationEvaluationStep.step_id,
        ValidationEvaluationOutput(
            metrics={"validation/ACC": 1.0},
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
    archive_current_iteration(config, iteration=0)
    update_best_iteration_snapshot(config, iteration=0)
    snapshot_dir = config.best_iteration_snapshot_dir
    test_predictions = snapshot_dir / "eval_predictions_test.csv"
    test_labels = snapshot_dir / "eval_predictions_test.numeric_labels.csv"
    test_metrics = snapshot_dir / "eval_predictions_test.metrics.json"
    test_predictions.write_text(
        "id,prediction,probability_1\nt-0,1,0.9\n",
        encoding="utf-8",
    )
    test_labels.write_text(
        "id,numeric_label\nt-0,1\n",
        encoding="utf-8",
    )
    test_metrics.write_text(json.dumps({"ACC": 1.0}), encoding="utf-8")

    inputs = gather_iteration_inputs(config=config, iteration=0)

    test_split = next(split for split in inputs.splits if split.split_name == "test")
    assert test_split.labeled_csv == test_labels
    assert test_split.preds_csv == test_predictions
    assert test_split.metrics == {"ACC": 1.0}
