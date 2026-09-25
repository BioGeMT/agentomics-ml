import json
from pathlib import Path

import pytest

from agentomics.datasets.dataset_preparation import (
    check_dataset_prepared,
    prepare_dataset,
    prepare_test_dataset,
)
from tests.dataset_helpers import write_raw_classification_split


def _snapshot_tree(root: Path) -> list[tuple[str, str, bytes | None, str | None]]:
    snapshot = []
    for path in sorted(
        (root, *root.rglob("*")),
        key=lambda entry: entry.relative_to(root).as_posix(),
    ):
        relative_path = path.relative_to(root).as_posix() or "."
        if path.is_symlink():
            snapshot.append((relative_path, "symlink", None, str(path.readlink())))
        elif path.is_dir():
            snapshot.append((relative_path, "directory", None, None))
        elif path.is_file():
            snapshot.append((relative_path, "file", path.read_bytes(), None))
        else:
            snapshot.append((relative_path, "other", None, None))
    return snapshot


def test_public_dataset_is_prepared_to_destination(
    default_dataset_with_validation: Path,
    tmp_path: Path,
):
    prepared_dir = tmp_path / "prepared" / default_dataset_with_validation.name

    prepare_dataset(
        source_dir=default_dataset_with_validation,
        destination_dir=prepared_dir,
    )

    assert (prepared_dir / "train" / "input" / "data.csv").is_file()
    assert check_dataset_prepared(prepared_dir)


def test_test_split_uses_public_label_mapping_without_modifying_source(
    default_dataset_with_validation_and_test: Path,
    tmp_path: Path,
):
    dataset_dir = default_dataset_with_validation_and_test
    prepared_dir = tmp_path / "prepared" / dataset_dir.name
    prepared_test_dir = tmp_path / "prepared_test" / dataset_dir.name
    train_metadata = prepare_dataset(
        source_dir=dataset_dir,
        destination_dir=prepared_dir,
    )
    source_snapshot = _snapshot_tree(dataset_dir)

    prepare_test_dataset(
        source_dir=dataset_dir,
        destination_dir=prepared_test_dir,
        task_type=train_metadata["task_type"],
        input_structure=train_metadata["input_structure"],
        label_to_scalar=train_metadata.get("label_to_scalar"),
    )

    test_labels_text = (prepared_test_dir / "test" / "labels.csv").read_text(
        encoding="utf-8"
    )
    test_metadata = json.loads(
        (prepared_test_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert "id,numeric_label" in test_labels_text
    assert test_metadata["prepared"]
    assert test_metadata["label_to_scalar"] == {"negative": 0, "positive": 1}
    assert _snapshot_tree(dataset_dir) == source_snapshot


def test_different_nested_files_with_matching_structure_are_prepared(tmp_path: Path):
    nested_dataset_dir = tmp_path / "nested_datasets" / "nested_dataset"
    write_raw_classification_split(
        nested_dataset_dir,
        "train",
        input_files={"images/train-0.png": "train image"},
        labels=("negative", "positive"),
    )
    write_raw_classification_split(
        nested_dataset_dir,
        "validation",
        input_files={"images/validation-0.png": "validation image"},
        labels=("negative", "positive"),
    )
    write_raw_classification_split(
        nested_dataset_dir,
        "test",
        input_files={"images/test-0.png": "test image"},
        labels=("negative", "positive"),
    )
    prepared_dir = tmp_path / "prepared_nested" / nested_dataset_dir.name
    prepared_test_dir = tmp_path / "prepared_nested_test" / nested_dataset_dir.name

    train_metadata = prepare_dataset(
        source_dir=nested_dataset_dir,
        destination_dir=prepared_dir,
        task_type="classification",
    )
    prepare_test_dataset(
        source_dir=nested_dataset_dir,
        destination_dir=prepared_test_dir,
        task_type=train_metadata["task_type"],
        input_structure=train_metadata["input_structure"],
        label_to_scalar=train_metadata.get("label_to_scalar"),
    )

    assert train_metadata["input_structure"] == ["images/"]
    assert (prepared_test_dir / "test" / "input" / "images").is_dir()


def test_unknown_test_label_is_rejected(
    default_dataset_with_validation_and_test: Path,
    tmp_path: Path,
):
    dataset_dir = default_dataset_with_validation_and_test
    train_metadata = prepare_dataset(
        source_dir=dataset_dir,
        destination_dir=tmp_path / "prepared" / dataset_dir.name,
    )
    (dataset_dir / "test" / "labels.csv").write_text(
        "id,label\ntest-0,unknown\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        prepare_test_dataset(
            source_dir=dataset_dir,
            destination_dir=tmp_path / "prepared_test" / dataset_dir.name,
            task_type=train_metadata["task_type"],
            input_structure=train_metadata["input_structure"],
            label_to_scalar=train_metadata.get("label_to_scalar"),
        )


def test_colocated_test_split_is_ignored_by_training_preparation(
    default_dataset_with_validation_and_test: Path,
    tmp_path: Path,
):
    dataset_dir = default_dataset_with_validation_and_test
    prepared_dir = tmp_path / "prepared" / dataset_dir.name

    prepare_dataset(
        source_dir=dataset_dir,
        destination_dir=prepared_dir,
    )

    assert check_dataset_prepared(prepared_dir)
    assert not (prepared_dir / "test").exists()


def test_symlinked_public_dataset_is_rejected(
    default_dataset_with_validation: Path,
    tmp_path: Path,
):
    prepared_dir = tmp_path / "prepared" / default_dataset_with_validation.name
    external_file = tmp_path / "external_reference.txt"
    external_file.write_text("external", encoding="utf-8")
    (
        default_dataset_with_validation / "train" / "input" / "linked.txt"
    ).symlink_to(external_file)

    with pytest.raises(ValueError):
        prepare_dataset(
            source_dir=default_dataset_with_validation,
            destination_dir=prepared_dir,
        )

    assert not check_dataset_prepared(prepared_dir)
