import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agentomics.runtime.collected_data import load_manifest
from agentomics.tools.download_tool import DownloadDeps, discard_file, download_file
from tests.helpers import serve_files

DATA = b"x" * 100


def _context(
    collected_data_directory: Path, max_bytes: int | None = None, allow_local_addresses: bool = True,
) -> RunContext[DownloadDeps]:
    """Create the context of a tool call; local addresses are allowed because the tests serve files locally."""
    deps = DownloadDeps(
        collected_data_directory=collected_data_directory,
        max_bytes=max_bytes,
        allow_local_addresses=allow_local_addresses,
    )
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _tar(members: dict[str, bytes], compression: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=f"w:{compression}") as bundle:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def _mark_first_member_encrypted(archive: bytes) -> bytes:
    """Set the encryption flag of the first member in the central directory of a zip archive."""
    flags_offset = archive.index(b"PK\x01\x02") + 8
    return archive[:flags_offset] + bytes([archive[flags_offset] | 0x1]) + archive[flags_offset + 1:]


def _saved_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_download_saves_file_and_records_it(tmp_path):
    with serve_files({"/data.csv": DATA}) as base_url:
        result = download_file(_context(tmp_path), f"{base_url}/data.csv", "data.csv")

    assert result.content_type == "text/csv"
    assert (tmp_path / "files" / "data.csv").read_bytes() == DATA
    assert json.loads((tmp_path / "manifest.json").read_text()) == [{
        "path": "files/data.csv",
        "source_url": f"{base_url}/data.csv",
        "size_bytes": 100,
        "sha256": hashlib.sha256(DATA).hexdigest(),
    }]


@pytest.mark.parametrize("send_content_length", [True, False], ids=["size-announced", "size-not-announced"])
def test_download_over_the_size_limit_is_refused(tmp_path, send_content_length):
    with serve_files({"/data.csv": DATA}, send_content_length=send_content_length) as base_url:
        result = download_file(_context(tmp_path, max_bytes=50), f"{base_url}/data.csv", "data.csv")

    assert "size limit" in result.error
    assert _saved_files(tmp_path) == []


def test_download_that_the_server_does_not_have_is_refused(tmp_path):
    with serve_files({}) as base_url:
        result = download_file(_context(tmp_path), f"{base_url}/missing.csv", "data.csv")

    assert "the download failed" in result.error
    assert _saved_files(tmp_path) == []


def test_download_from_a_local_address_is_refused_unless_allowed(tmp_path):
    with serve_files({"/data.csv": DATA}) as base_url:
        result = download_file(_context(tmp_path, allow_local_addresses=False), f"{base_url}/data.csv", "data.csv")

    assert "not a public address" in result.error
    assert _saved_files(tmp_path) == []


def test_url_that_cannot_be_parsed_is_refused_without_ending_the_run(tmp_path):
    result = download_file(_context(tmp_path), "http://example.org:port/data.csv", "data.csv")

    assert "the download failed" in result.error


@pytest.mark.parametrize(
    ("filename", "reason"),
    [("../data.csv", "not a plain file name"), ("existing.csv", "already exists")],
)
def test_unsafe_or_existing_file_name_is_refused(tmp_path, filename, reason):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "existing.csv").write_bytes(b"old")

    with serve_files({"/data.csv": DATA}) as base_url:
        result = download_file(_context(tmp_path), f"{base_url}/data.csv", filename)

    assert reason in result.error
    assert _saved_files(tmp_path) == ["files/existing.csv"]
    assert (tmp_path / "files" / "existing.csv").read_bytes() == b"old"


def test_rerun_counts_the_files_of_an_earlier_run_against_the_limit(tmp_path):
    with serve_files({"/a.csv": b"a" * 100, "/big.csv": b"c" * 200}) as base_url:
        download_file(_context(tmp_path, max_bytes=250), f"{base_url}/a.csv", "a.csv")

        rerun_result = download_file(_context(tmp_path, max_bytes=250), f"{base_url}/big.csv", "big.csv")

    assert "size limit" in rerun_result.error
    assert rerun_result.budget.used_bytes == 100


def test_rerun_can_discard_a_file_of_an_earlier_run(tmp_path):
    with serve_files({"/a.csv": b"a" * 100}) as base_url:
        download_file(_context(tmp_path), f"{base_url}/a.csv", "a.csv")

    discard_file(_context(tmp_path), "files/a.csv")

    assert _saved_files(tmp_path) == ["manifest.json"]
    assert load_manifest(tmp_path) == []


def test_discarding_a_path_that_was_not_downloaded_says_it_is_not_in_the_manifest(tmp_path):
    result = discard_file(_context(tmp_path), "files/missing.csv")

    assert "not in the manifest" in result.error


@pytest.mark.parametrize(
    ("filename", "archive", "unpacked_paths"),
    [
        ("data.zip", _zip({"a.csv": DATA, ".hidden.csv": DATA, "../escape.csv": DATA}), ["files/data/a.csv", "files/data/.hidden.csv"]),
        ("data.tar.gz", _tar({"a.csv": DATA, ".hidden.csv": DATA, "../escape.csv": DATA}, "gz"), ["files/data/a.csv", "files/data/.hidden.csv"]),
        ("data.tar.bz2", _tar({"a.csv": DATA}, "bz2"), ["files/data/a.csv"]),
        ("data.csv.gz", gzip.compress(DATA), ["files/data.csv"]),
    ],
    ids=["zip", "tar.gz", "tar.bz2", "gz"],
)
def test_archive_is_unpacked_into_recorded_files(tmp_path, filename, archive, unpacked_paths):
    with serve_files({f"/{filename}": archive}) as base_url:
        download_file(_context(tmp_path), f"{base_url}/{filename}", filename)

    entries = load_manifest(tmp_path)
    assert [entry.path for entry in entries] == unpacked_paths
    assert all(entry.archive == filename for entry in entries)
    assert _saved_files(tmp_path) == sorted([*unpacked_paths, "manifest.json"])
    assert (tmp_path / unpacked_paths[0]).read_bytes() == DATA


def test_discarding_the_folder_of_an_unpacked_archive_removes_all_its_files(tmp_path):
    context = _context(tmp_path)
    with serve_files({"/data.zip": _zip({"a.csv": DATA, "b.csv": DATA})}) as base_url:
        download_file(context, f"{base_url}/data.zip", "data.zip")

    discard_file(context, "files/data")

    assert _saved_files(tmp_path) == ["manifest.json"]
    assert not (tmp_path / "files" / "data").exists()
    assert load_manifest(tmp_path) == []


@pytest.mark.parametrize(
    ("archive", "reason"),
    [
        (b"not a zip archive", "could not be unpacked"),
        (_mark_first_member_encrypted(_zip({"a.csv": DATA})), "encrypted"),
        (_zip({"a.csv": b"x" * 1000, "b.csv": b"x" * 1000}), "size limit"),
    ],
    ids=["unreadable", "encrypted", "unpacked-files-over-the-size-limit"],
)
def test_archive_that_cannot_be_unpacked_within_the_limit_is_not_kept(tmp_path, archive, reason):
    with serve_files({"/data.zip": archive}) as base_url:
        result = download_file(_context(tmp_path, max_bytes=1500), f"{base_url}/data.zip", "data.zip")

    assert reason in result.error
    assert _saved_files(tmp_path) == []
