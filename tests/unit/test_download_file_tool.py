import gzip
import hashlib
import json

import pytest

from agentomics.runtime.collected_data_directory import load_manifest
from tests.helpers import create_download_context, create_zip_archive, list_files, mock_downloads

DATA = b"x" * 100
URL = "https://example.org"


def test_download_saves_file_and_records_it(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/data.csv": DATA})
    context = create_download_context(data_module_config)

    result = download_file_tool.function(context, f"{URL}/data.csv", "data.csv")

    assert (result.path, result.file_count, result.content_type) == ("files/data.csv", 1, "text/csv")
    assert (data_module_config.files_dir / "data.csv").read_bytes() == DATA
    assert json.loads(data_module_config.manifest_path.read_text(encoding="utf-8")) == [{
        "path": "files/data.csv",
        "source_url": f"{URL}/data.csv",
        "size_bytes": 100,
        "sha256": hashlib.sha256(DATA).hexdigest(),
    }]


@pytest.mark.parametrize("send_content_length", [True, False], ids=["size-announced", "size-not-announced"])
def test_download_over_the_size_limit_is_refused(data_module_config, download_file_tool, monkeypatch, send_content_length):
    mock_downloads(monkeypatch, {f"{URL}/data.csv": DATA}, send_content_length=send_content_length)
    context = create_download_context(data_module_config, max_bytes=50)

    result = download_file_tool.function(context, f"{URL}/data.csv", "data.csv")

    assert result.startswith("Error:") and "size limit" in result
    assert list_files(data_module_config.collected_data_dir) == ["manifest.json"]


def test_download_counts_against_the_byte_budget_of_later_downloads(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/a.csv": b"a" * 100, f"{URL}/big.csv": b"c" * 200})
    context = create_download_context(data_module_config, max_bytes=250)

    first_result = download_file_tool.function(context, f"{URL}/a.csv", "a.csv")
    second_result = download_file_tool.function(context, f"{URL}/big.csv", "big.csv")

    assert (first_result.used_bytes, first_result.max_bytes) == (100, 250)
    assert "size limit" in second_result
    assert context.deps.byte_budget.used_bytes == 100


@pytest.mark.parametrize(
    "url",
    [f"{URL}/missing.csv", "http://example.org:port/data.csv", "http://example..com/data.csv", "ftp://example.org/data.csv"],
    ids=["file-the-server-does-not-have", "port-that-is-not-a-number", "host-with-an-empty-label", "url-that-is-not-http"],
)
def test_url_that_cannot_be_downloaded_is_refused_without_ending_the_run(data_module_config, download_file_tool, monkeypatch, url):
    mock_downloads(monkeypatch, {})

    result = download_file_tool.function(create_download_context(data_module_config), url, "data.csv")

    assert result.startswith("Error:")
    assert list_files(data_module_config.collected_data_dir) == ["manifest.json"]


@pytest.mark.parametrize(
    ("filename", "reason"),
    [("../data.csv", "not a plain file name"), ("..zip", "not a plain file name"), ("existing.csv", "already exists")],
)
def test_unsafe_or_existing_file_name_is_refused(data_module_config, download_file_tool, monkeypatch, filename, reason):
    mock_downloads(monkeypatch, {f"{URL}/data.csv": DATA, f"{URL}/existing.csv": b"old"})
    context = create_download_context(data_module_config)
    download_file_tool.function(context, f"{URL}/existing.csv", "existing.csv")

    result = download_file_tool.function(context, f"{URL}/data.csv", filename)

    assert reason in result
    assert (data_module_config.files_dir / "existing.csv").read_bytes() == b"old"
    assert list_files(data_module_config.collected_data_dir) == ["files/existing.csv", "manifest.json"]


def test_archive_is_unpacked_into_recorded_files_and_removed(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/data.zip": create_zip_archive([("a.csv", DATA), ("sub/b.csv", DATA)])})

    result = download_file_tool.function(create_download_context(data_module_config), f"{URL}/data.zip", "data.zip")

    entries = load_manifest(data_module_config)
    assert (result.path, result.file_count) == ("files/data", 2)
    assert [entry.path for entry in entries] == ["files/data/a.csv", "files/data/sub/b.csv"]
    assert all(entry.archive == "data.zip" and entry.source_url == f"{URL}/data.zip" for entry in entries)
    assert list_files(data_module_config.collected_data_dir) == ["files/data/a.csv", "files/data/sub/b.csv", "manifest.json"]


def test_gz_file_is_unpacked_into_one_file_named_without_the_suffix(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/data.csv.gz": gzip.compress(DATA)})

    download_file_tool.function(create_download_context(data_module_config), f"{URL}/data.csv.gz", "data.csv.gz")

    assert [entry.path for entry in load_manifest(data_module_config)] == ["files/data.csv"]
    assert (data_module_config.files_dir / "data.csv").read_bytes() == DATA


def test_file_that_an_archive_holds_twice_counts_once_against_the_byte_budget(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/data.zip": create_zip_archive([("a.csv", b"1" * 10), ("a.csv", b"2" * 30)])})
    context = create_download_context(data_module_config)

    result = download_file_tool.function(context, f"{URL}/data.zip", "data.zip")

    assert result.file_count == 1
    assert context.deps.byte_budget.used_bytes == 30
    assert (data_module_config.files_dir / "data" / "a.csv").read_bytes() == b"2" * 30


@pytest.mark.parametrize(
    ("archive", "reason"),
    [
        (b"not a zip archive", "could not be unpacked"),
        (create_zip_archive([("a.csv", b"x" * 1000), ("b.csv", b"x" * 1000)]), "size limit"),
    ],
    ids=["unreadable", "unpacked-files-over-the-size-limit"],
)
def test_archive_that_cannot_be_unpacked_is_not_kept(data_module_config, download_file_tool, monkeypatch, archive, reason):
    mock_downloads(monkeypatch, {f"{URL}/data.zip": archive})
    context = create_download_context(data_module_config, max_bytes=1500)

    result = download_file_tool.function(context, f"{URL}/data.zip", "data.zip")

    assert reason in result
    assert context.deps.byte_budget.used_bytes == 0
    assert list_files(data_module_config.collected_data_dir) == ["manifest.json"]
