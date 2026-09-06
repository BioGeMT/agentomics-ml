from pathlib import Path

import pandas as pd
import pytest

from agentomics.datasets.csv_converter import convert_csv_dataset


def test_converter_writes_supplied_test_split(tmp_path: Path):
    output_dir = tmp_path / "datasets" / "toy"

    convert_csv_dataset(
        output_dir=output_dir,
        label_column="target",
        splits={
            "train": pd.DataFrame(
                {"id": ["train-0"], "feature": [1], "target": ["a"]}
            ),
            "test": pd.DataFrame(
                {"id": ["test-0"], "feature": [2], "target": ["a"]}
            ),
        },
        task_type="classification",
    )

    assert (output_dir / "test" / "input" / "data.csv").is_file()
    assert (output_dir / "test" / "labels.csv").is_file()
    assert (output_dir / "metadata.json").is_file()


def test_converter_omits_test_folder_when_no_test_split(tmp_path: Path):
    output_dir = tmp_path / "datasets" / "toy"

    convert_csv_dataset(
        output_dir=output_dir,
        label_column="target",
        splits={
            "train": pd.DataFrame(
                {"id": ["train-0"], "feature": [1], "target": ["a"]}
            ),
        },
    )

    assert (output_dir / "train" / "input" / "data.csv").is_file()
    assert not (output_dir / "test").exists()


def test_converter_requires_train_split(tmp_path: Path):
    with pytest.raises(ValueError):
        convert_csv_dataset(
            output_dir=tmp_path / "datasets" / "toy",
            label_column="target",
            splits={
                "test": pd.DataFrame(
                    {"id": ["test-0"], "feature": [2], "target": ["a"]}
                ),
            },
        )
