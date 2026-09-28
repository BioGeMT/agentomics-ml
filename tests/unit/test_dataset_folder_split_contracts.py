import json
from pathlib import Path

import pytest

from agentomics.datasets.data_contract import (
    NUMERIC_LABEL_COLUMN_NAME,
    validate_and_read_labels,
)
from agentomics.datasets.dataset_preparation import (
    check_dataset_prepared,
    prepare_dataset,
)
from agentomics.datasets.datasets_interactive import get_all_datasets_info


def test_prepared_folder_dataset_satisfies_dataset_contract(
    default_prepared_dataset_dir: Path,
):
    assert check_dataset_prepared(default_prepared_dataset_dir)


def test_dataset_listing_counts_folder_labels_instead_of_stale_metadata(
    default_prepared_dataset_dir: Path,
):
    metadata_path = default_prepared_dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["splits"] = {"train_rows": 999, "validation_rows": 999}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    [info] = get_all_datasets_info(default_prepared_dataset_dir.parent)

    assert info["name"] == default_prepared_dataset_dir.name
    assert info["status"] == "Prepared"
    assert info["train_rows"] == 2
    assert info["validation_rows"] == 2


def test_preparation_rejects_non_numeric_regression_labels(
    default_dataset_dir: Path,
    tmp_path: Path,
):
    (default_dataset_dir / "metadata.json").unlink()
    (default_dataset_dir / "train" / "labels.csv").write_text(
        "id,label\ntrain-0,not-a-number\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        prepare_dataset(
            source_dir=default_dataset_dir,
            destination_dir=tmp_path / "prepared",
            task_type="regression",
        )


def test_labels_csv_rejects_extra_columns(tmp_path: Path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "id,numeric_label,extra\nsample-1,0,unused\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_and_read_labels(
            labels_path,
            NUMERIC_LABEL_COLUMN_NAME,
            require_numeric_values=True,
        )


def test_preparation_derives_class_mapping_from_folder_labels(
    default_dataset_dir: Path,
    tmp_path: Path,
):
    prepared_dir = tmp_path / "prepared"

    prepare_dataset(
        source_dir=default_dataset_dir,
        destination_dir=prepared_dir,
        task_type="classification",
    )

    metadata = json.loads(
        (prepared_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["label_to_scalar"] == {"negative": 0, "positive": 1}


def test_preparation_preserves_numeric_looking_ids(
    default_dataset_dir: Path,
    tmp_path: Path,
):
    (default_dataset_dir / "train" / "labels.csv").write_text(
        "id,label\n001,positive\n002,negative\n",
        encoding="utf-8",
    )
    prepared_dir = tmp_path / "prepared"

    prepare_dataset(
        source_dir=default_dataset_dir,
        destination_dir=prepared_dir,
        task_type="classification",
    )

    prepared_labels = (
        (prepared_dir / "train" / "labels.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert prepared_labels == ["id,numeric_label", "001,1", "002,0"]


def test_preparation_leaves_absent_validation_split_absent(
    default_dataset_dir: Path,
    tmp_path: Path,
):
    prepared_dir = tmp_path / "prepared"

    prepare_dataset(
        source_dir=default_dataset_dir,
        destination_dir=prepared_dir,
        task_type="classification",
    )

    assert (prepared_dir / "train").is_dir()
    assert not (prepared_dir / "validation").exists()
