import pytest

from agentomics.runtime.collected_data_directory import (
    move_into_collected_data_directory,
    validate_collected_data_directory,
)


def _add_file(config) -> None:
    """Add one file to the collected data folder, recorded in its manifest."""
    path = config.collected_data_dir / "downloaded.csv"
    path.write_bytes(b"x" * 100)
    move_into_collected_data_directory(config, path, "data.csv", "https://example.org/data.csv", source_archive=None)


def test_folder_that_matches_its_manifest_is_valid(initialized_data_module_config):
    _add_file(initialized_data_module_config)

    validate_collected_data_directory(initialized_data_module_config)


def test_recorded_file_that_is_missing_makes_the_folder_invalid(initialized_data_module_config):
    _add_file(initialized_data_module_config)
    (initialized_data_module_config.files_dir / "data.csv").unlink()

    with pytest.raises(ValueError, match="files/data.csv is missing"):
        validate_collected_data_directory(initialized_data_module_config)


def test_file_that_the_manifest_does_not_record_makes_the_folder_invalid(initialized_data_module_config):
    _add_file(initialized_data_module_config)
    (initialized_data_module_config.files_dir / "extra.csv").write_bytes(b"extra")

    with pytest.raises(ValueError, match="files/extra.csv is not in the manifest"):
        validate_collected_data_directory(initialized_data_module_config)


def test_file_that_fails_to_be_hashed_is_not_moved_into_the_folder(initialized_data_module_config, monkeypatch):
    def fail_to_hash(path):
        raise OSError("interrupted")

    monkeypatch.setattr("agentomics.runtime.collected_data_directory.get_file_hash", fail_to_hash)

    with pytest.raises(OSError, match="interrupted"):
        _add_file(initialized_data_module_config)

    assert not (initialized_data_module_config.files_dir / "data.csv").exists()
    validate_collected_data_directory(initialized_data_module_config)
