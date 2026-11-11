import pytest

from agentomics.runtime.collected_data_directory import (
    add_to_collected_data_directory,
    validate_collected_data_directory,
)


def _add_file(config) -> None:
    """Add one file to the collected data folder, recorded in its manifest."""
    path = config.collected_data_dir / "downloaded.csv"
    path.write_bytes(b"x" * 100)
    add_to_collected_data_directory(config, path, "data.csv", "https://example.org/data.csv", archive=None)


def test_folder_that_matches_its_manifest_is_valid(data_module_config):
    _add_file(data_module_config)

    validate_collected_data_directory(data_module_config)


def test_recorded_file_that_is_missing_makes_the_folder_invalid(data_module_config):
    _add_file(data_module_config)
    (data_module_config.files_dir / "data.csv").unlink()

    with pytest.raises(ValueError, match="files/data.csv is missing"):
        validate_collected_data_directory(data_module_config)


def test_file_that_the_manifest_does_not_record_makes_the_folder_invalid(data_module_config):
    _add_file(data_module_config)
    (data_module_config.files_dir / "extra.csv").write_bytes(b"extra")

    with pytest.raises(ValueError, match="files/extra.csv is not in the manifest"):
        validate_collected_data_directory(data_module_config)


def test_folder_of_a_finished_data_collection_is_invalid(data_module_config):
    data_module_config.output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="finished data collection"):
        validate_collected_data_directory(data_module_config)
