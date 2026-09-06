import json
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from agentomics.agents.steps.data_split import DataSplitOutput, DataSplitStep
from agentomics.agents.steps.validation_evaluation import (
    ValidationEvaluationOutput,
    ValidationEvaluationStep,
)
from agentomics.runtime import best_iteration_snapshot
from agentomics.runtime.conda_utils import (
    get_iteration_environment_archive_path,
    restore_iteration_environment,
)
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import save_step_output
from agentomics.utils.config import Config
from tests.helpers import create_runtime_step, write_split_folder


def _create_archived_iteration(
    config: Config,
    *,
    iteration: int = 1,
    is_new_best: bool,
    split_changed: bool,
    model_artifact_path: Path | None = None,
) -> Path:
    prepare_iteration_workspace(config, iteration=iteration)
    split_dir = config.splits_dir / "split_1"
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
            split_changed=split_changed,
            split_version=1,
        ),
    )
    create_runtime_step(DataSplitStep, config, model=Mock()).archive_step_folder()
    save_step_output(
        config,
        ValidationEvaluationStep.step_id,
        ValidationEvaluationOutput(
            metrics={"validation/ACC": 0.9},
            is_new_best=is_new_best,
            status="success",
        ),
    )
    create_runtime_step(
        ValidationEvaluationStep,
        config,
        model=Mock(),
    ).archive_step_folder()
    if model_artifact_path is not None:
        model_artifact = config.current_iteration_dir / model_artifact_path
        model_artifact.parent.mkdir(parents=True, exist_ok=True)
        model_artifact.write_bytes(b"trained model")
    update_current_iteration_state(config, status="success", ended_at=time.time())
    archive_current_iteration(config, iteration)
    return config.iteration_dir(iteration)


def _write_previous_snapshot_marker(config: Config) -> Path:
    marker = config.best_iteration_snapshot_dir / "previous-model.bin"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("previous", encoding="utf-8")
    return marker


def test_new_best_iteration_replaces_snapshot_with_archived_iteration(
    initialized_run_with_lightweight_environment: Config,
):
    config = initialized_run_with_lightweight_environment
    config.conda_export_mode = "yaml"
    previous_marker = _write_previous_snapshot_marker(config)
    model_artifact_path = Path("model_training/training_artifacts/model.bin")
    iteration_dir = _create_archived_iteration(
        config,
        is_new_best=True,
        split_changed=True,
        model_artifact_path=model_artifact_path,
    )
    model_artifact = iteration_dir / model_artifact_path

    best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    snapshot_dir = config.best_iteration_snapshot_dir
    assert not previous_marker.exists()
    assert (
        snapshot_dir / "model_training" / "training_artifacts" / "model.bin"
    ).read_bytes() == b"trained model"
    assert model_artifact.read_bytes() == b"trained model"
    metadata = json.loads(
        (
            snapshot_dir
            / Config.RUNTIME_INFO_DIRNAME
            / Config.ITERATION_METADATA_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert metadata["iteration"] == 1


def test_snapshot_contains_descriptor_and_excludes_transient_content(
    initialized_run_with_lightweight_environment: Config,
):
    config = initialized_run_with_lightweight_environment
    config.conda_export_mode = "yaml"
    iteration_dir = _create_archived_iteration(
        config,
        is_new_best=True,
        split_changed=False,
    )
    for junk_name in (".cache", ".conda", "__pycache__"):
        junk_dir = iteration_dir / junk_name
        junk_dir.mkdir()
        (junk_dir / "temporary.bin").write_bytes(b"temporary")
    (iteration_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME).write_text(
        "stale descriptor",
        encoding="utf-8",
    )

    best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    snapshot_dir = config.best_iteration_snapshot_dir
    descriptor = (
        snapshot_dir
        / Config.RUNTIME_INFO_DIRNAME
        / Config.ENVIRONMENT_DESCRIPTOR_FILENAME
    )
    assert descriptor.is_file()
    assert "channels:" in descriptor.read_text(encoding="utf-8")
    assert not (snapshot_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME).exists()
    assert all(
        not (snapshot_dir / name).exists()
        for name in (".cache", ".conda", "__pycache__")
    )
    assert not get_iteration_environment_archive_path(snapshot_dir).exists()


def test_full_export_mode_includes_environment_archive(
    initialized_run_with_lightweight_environment: Config,
    monkeypatch,
):
    config = initialized_run_with_lightweight_environment
    config.conda_export_mode = "full"
    _create_archived_iteration(config, is_new_best=True, split_changed=False)

    def create_archive(_environment_path: Path, archive_path: Path) -> None:
        archive_path.write_bytes(b"packed environment")

    monkeypatch.setattr(
        best_iteration_snapshot,
        "export_environment_archive",
        create_archive,
    )

    best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    snapshot_dir = config.best_iteration_snapshot_dir
    descriptor_path = (
        snapshot_dir
        / Config.RUNTIME_INFO_DIRNAME
        / Config.ENVIRONMENT_DESCRIPTOR_FILENAME
    )
    archive_path = get_iteration_environment_archive_path(snapshot_dir)
    assert descriptor_path.is_file()
    assert archive_path.read_bytes() == b"packed environment"


def test_failed_environment_packing_leaves_usable_snapshot(
    initialized_run_with_lightweight_environment: Config,
    monkeypatch,
    capsys,
    tmp_path: Path,
):
    config = initialized_run_with_lightweight_environment
    config.conda_export_mode = "full"
    _create_archived_iteration(
        config,
        is_new_best=True,
        split_changed=False,
        model_artifact_path=Path("model.bin"),
    )

    def fail_after_partial_archive(
        _environment_path: Path,
        archive_path: Path,
    ) -> None:
        archive_path.write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, "conda-pack")

    monkeypatch.setattr(
        best_iteration_snapshot,
        "export_environment_archive",
        fail_after_partial_archive,
    )

    best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    snapshot_dir = config.best_iteration_snapshot_dir
    assert (snapshot_dir / "model.bin").read_bytes() == b"trained model"
    assert (
        snapshot_dir
        / Config.RUNTIME_INFO_DIRNAME
        / Config.ENVIRONMENT_DESCRIPTOR_FILENAME
    ).is_file()
    assert not get_iteration_environment_archive_path(snapshot_dir).exists()
    restored_environment = restore_iteration_environment(
        snapshot_dir,
        tmp_path / "restored-environment",
    )
    subprocess.run(
        [str(restored_environment / "bin" / "python"), "-c", "import yaml"],
        check=True,
    )
    assert "environment will be rebuilt from environment.yml" in capsys.readouterr().out


def test_missing_environment_preserves_previous_snapshot(
    initialized_run_config: Config,
):
    config = initialized_run_config
    config.conda_export_mode = "yaml"
    previous_marker = _write_previous_snapshot_marker(config)
    _create_archived_iteration(config, is_new_best=True, split_changed=False)

    with pytest.raises(FileNotFoundError, match="shared Conda environment is missing"):
        best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    assert previous_marker.read_text(encoding="utf-8") == "previous"


def test_split_change_without_replacement_removes_previous_snapshot(
    initialized_run_config: Config,
):
    config = initialized_run_config
    config.conda_export_mode = "yaml"
    _write_previous_snapshot_marker(config)
    _create_archived_iteration(config, is_new_best=False, split_changed=True)

    best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    assert not config.best_iteration_snapshot_dir.exists()


def test_non_best_iteration_on_same_split_preserves_previous_snapshot(
    initialized_run_config: Config,
):
    config = initialized_run_config
    config.conda_export_mode = "yaml"
    previous_marker = _write_previous_snapshot_marker(config)
    _create_archived_iteration(config, is_new_best=False, split_changed=False)

    best_iteration_snapshot.update_best_iteration_snapshot(config, iteration=1)

    assert previous_marker.read_text(encoding="utf-8") == "previous"
