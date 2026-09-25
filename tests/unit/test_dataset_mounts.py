from pathlib import Path

import pytest

from agentomics.cli.run import build_dataset_mounts


@pytest.mark.parametrize(
    "entry_name",
    [
        pytest.param("train", id="train-directory"),
        pytest.param("validation", id="validation-directory"),
        pytest.param("supplementary", id="supplementary-directory"),
        pytest.param("train.csv", id="train-csv"),
        pytest.param("validation.csv", id="validation-csv"),
        pytest.param("metadata.json", id="metadata"),
        pytest.param("dataset_description.md", id="description"),
    ],
)
def test_public_dataset_entry_is_mounted_readonly(tmp_path: Path, entry_name: str):
    dataset_directory = tmp_path / "toy"
    dataset_directory.mkdir()
    entry = dataset_directory / entry_name
    if entry.suffix:
        entry.write_text("data", encoding="utf-8")
    else:
        entry.mkdir()

    mounts = build_dataset_mounts(dataset_directory)

    assert mounts == [
        "type=tmpfs,dst=/datasets",
        (
            f"type=bind,src={entry.resolve()},"
            f"dst=/datasets/{dataset_directory.name}/{entry_name},readonly"
        ),
    ]


@pytest.mark.parametrize(
    "entry_name",
    [
        pytest.param("test", id="test-directory"),
        pytest.param("test_external", id="named-test-directory"),
        pytest.param("test.csv", id="test-csv"),
    ],
)
def test_test_dataset_entry_is_not_mounted(tmp_path: Path, entry_name: str):
    dataset_directory = tmp_path / "toy"
    dataset_directory.mkdir()
    entry = dataset_directory / entry_name
    if entry.suffix:
        entry.write_text("data", encoding="utf-8")
    else:
        entry.mkdir()

    assert build_dataset_mounts(dataset_directory) == ["type=tmpfs,dst=/datasets"]


def test_symlinked_dataset_entry_is_rejected(tmp_path: Path):
    dataset_directory = tmp_path / "toy"
    dataset_directory.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (dataset_directory / "validation").symlink_to(external)

    with pytest.raises(ValueError):
        build_dataset_mounts(dataset_directory)
