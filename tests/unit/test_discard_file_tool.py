from agentomics.runtime.collected_data_directory import load_manifest
from tests.helpers import create_download_context, create_zip_archive, mock_downloads

DATA = b"x" * 100
URL = "https://example.org"


def test_discard_removes_the_file_and_its_manifest_entry_and_releases_its_bytes(
    data_module_config, download_file_tool, discard_file_tool, monkeypatch,
):
    mock_downloads(monkeypatch, {f"{URL}/a.csv": DATA, f"{URL}/b.csv": DATA})
    context = create_download_context(data_module_config)
    download_file_tool.function(context, f"{URL}/a.csv", "a.csv")
    download_file_tool.function(context, f"{URL}/b.csv", "b.csv")

    result = discard_file_tool.function(context, "files/a.csv")

    assert (result.file_count, result.used_bytes) == (1, 100)
    assert [entry.path for entry in load_manifest(data_module_config)] == ["files/b.csv"]
    assert not (data_module_config.files_dir / "a.csv").exists()


def test_discarding_the_folder_of_an_unpacked_archive_removes_all_its_files(
    data_module_config, download_file_tool, discard_file_tool, monkeypatch,
):
    mock_downloads(monkeypatch, {f"{URL}/data.zip": create_zip_archive([("a.csv", DATA), ("b.csv", DATA)])})
    context = create_download_context(data_module_config)
    download_file_tool.function(context, f"{URL}/data.zip", "data.zip")

    result = discard_file_tool.function(context, "files/data")

    assert (result.file_count, result.used_bytes) == (2, 0)
    assert load_manifest(data_module_config) == []
    assert not (data_module_config.files_dir / "data").exists()


def test_discarding_the_last_file_of_an_unpacked_archive_removes_its_folder(
    data_module_config, download_file_tool, discard_file_tool, monkeypatch,
):
    mock_downloads(monkeypatch, {f"{URL}/data.zip": create_zip_archive([("sub/a.csv", DATA)])})
    context = create_download_context(data_module_config)
    download_file_tool.function(context, f"{URL}/data.zip", "data.zip")

    discard_file_tool.function(context, "files/data/sub/a.csv")

    assert not (data_module_config.files_dir / "data").exists()
    assert data_module_config.files_dir.is_dir()


def test_discarding_a_path_that_was_not_downloaded_says_it_is_not_in_the_manifest(data_module_config, discard_file_tool):
    result = discard_file_tool.function(create_download_context(data_module_config), "files/missing.csv")

    assert result.startswith("Error:") and "not in the manifest" in result
