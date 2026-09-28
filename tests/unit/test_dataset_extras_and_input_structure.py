import json
from pathlib import Path

import pytest

from agentomics.datasets.data_contract import (
    record_input_dir_structure,
    validate_splits,
)
from agentomics.datasets.dataset_preparation import prepare_dataset


def test_input_structure_records_only_top_level_files_and_directories(tmp_path: Path):
    input_dir = tmp_path / "input"
    images_dir = input_dir / "images"
    images_dir.mkdir(parents=True)
    (input_dir / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (images_dir / "sample.png").write_text("image", encoding="utf-8")

    structure = record_input_dir_structure(input_dir)

    assert structure == ["data.csv", "images/"]


def test_split_contract_accepts_symlinked_samples_in_matching_directory(
    tmp_path: Path,
):
    source_audio = tmp_path / "source" / "validation.wav"
    source_audio.parent.mkdir()
    source_audio.write_text("validation audio", encoding="utf-8")
    validation_split = tmp_path / "validation"
    validation_audio = validation_split / "input" / "audio"
    validation_audio.mkdir(parents=True)
    (validation_audio / "validation.wav").symlink_to(source_audio)
    (validation_split / "labels.csv").write_text(
        "id,numeric_label\nvalidation,0\n",
        encoding="utf-8",
    )

    validate_splits(
        {"validation": validation_split},
        expected_input_structure=["audio/"],
    )


def test_split_contract_distinguishes_file_from_directory(tmp_path: Path):
    validation_split = tmp_path / "validation"
    (validation_split / "input" / "data").mkdir(parents=True)
    (validation_split / "labels.csv").write_text(
        "id,numeric_label\nvalidation-0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_splits(
            {"validation": validation_split},
            expected_input_structure=["data"],
        )


@pytest.mark.parametrize(
    "input_change",
    [
        pytest.param("missing", id="missing-input"),
        pytest.param("extra", id="extra-input"),
    ],
)
def test_preparation_rejects_validation_with_different_input_structure(
    default_dataset_with_validation: Path,
    tmp_path: Path,
    input_change: str,
):
    validation_input_dir = default_dataset_with_validation / "validation" / "input"
    if input_change == "missing":
        (validation_input_dir / "data.csv").unlink()
    else:
        (validation_input_dir / "unexpected.csv").write_text(
            "id,other\nvalidation-0,value\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError):
        prepare_dataset(
            source_dir=default_dataset_with_validation,
            destination_dir=tmp_path / "prepared",
            task_type="classification",
        )


def test_preparation_rejects_supplementary_content_inside_split(
    default_dataset_dir: Path,
    tmp_path: Path,
):
    supplementary_dir = default_dataset_dir / "train" / "supplementary"
    supplementary_dir.mkdir()
    (supplementary_dir / "augmented.csv").write_text(
        "augmented data",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        prepare_dataset(
            source_dir=default_dataset_dir,
            destination_dir=tmp_path / "prepared",
            task_type="classification",
        )


def test_preparation_records_input_structure_in_metadata(
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
    assert metadata["input_structure"] == ["data.csv"]


def test_preparation_preserves_top_level_supplementary_content(
    default_dataset_dir: Path,
    tmp_path: Path,
):
    prepared_dir = tmp_path / "prepared"

    prepare_dataset(
        source_dir=default_dataset_dir,
        destination_dir=prepared_dir,
        task_type="classification",
    )

    source_supplementary = default_dataset_dir / "supplementary"
    prepared_supplementary = prepared_dir / "supplementary"
    source_files = {
        path.relative_to(source_supplementary): path.read_bytes()
        for path in source_supplementary.rglob("*")
        if path.is_file()
    }
    prepared_files = {
        path.relative_to(prepared_supplementary): path.read_bytes()
        for path in prepared_supplementary.rglob("*")
        if path.is_file()
    }
    assert prepared_files == source_files
