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
from agentomics.tools.download_tool import DownloadDeps, check_collected_data, current_budget, discard_file, download_file
from tests.helpers import serve_files

DATA = b"x" * 100


def _context(collected_data_directory: Path, max_bytes: int | None = None) -> RunContext[DownloadDeps]:
    deps = DownloadDeps(collected_data_directory=collected_data_directory, max_bytes=max_bytes)
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def _saved_files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_download_saves_file_and_records_it(tmp_path):
    with serve_files({"/data.csv": DATA}) as base_url:
        result = download_file(_context(tmp_path), f"{base_url}/data.csv", "data.csv")

    assert "text/csv" in result
    assert (tmp_path / "files" / "data.csv").read_bytes() == DATA
    assert json.loads((tmp_path / "manifest.json").read_text()) == [{
        "path": "data.csv",
        "source_url": f"{base_url}/data.csv",
        "size_bytes": 100,
        "sha256": hashlib.sha256(DATA).hexdigest(),
    }]


@pytest.mark.parametrize(
    ("url_path", "limit_bytes"),
    [
        ("/nolength/data.csv", 50),  # the streamed bytes go over the limit
        ("/missing.csv", None),  # the server answers 404
    ],
)
def test_rejected_download_records_nothing(tmp_path, url_path, limit_bytes):
    with serve_files({"/data.csv": DATA}) as base_url:
        result = download_file(_context(tmp_path, limit_bytes), f"{base_url}{url_path}", "data.csv")

    assert result.startswith("Error")
    assert _saved_files(tmp_path) == []


@pytest.mark.parametrize("filename", ["../data.csv", "existing.csv"])
def test_unsafe_or_existing_file_name_is_refused(tmp_path, filename):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "existing.csv").write_bytes(b"old")

    with serve_files({"/data.csv": DATA}) as base_url:
        result = download_file(_context(tmp_path), f"{base_url}/data.csv", filename)

    assert result.startswith("Error")
    assert _saved_files(tmp_path) == ["files/existing.csv"]
    assert (tmp_path / "files" / "existing.csv").read_bytes() == b"old"


def _change_bytes(files_directory: Path):
    (files_directory / "data.csv").write_bytes(b"y" * 100)


def _remove_file(files_directory: Path):
    (files_directory / "data.csv").unlink()


def _add_unrecorded_file(files_directory: Path):
    (files_directory / "extra.csv").write_bytes(b"extra")


@pytest.mark.parametrize("change", [_change_bytes, _remove_file, _add_unrecorded_file])
def test_check_reports_files_that_disagree_with_the_manifest(tmp_path, change):
    context = _context(tmp_path)
    with serve_files({"/data.csv": DATA}) as base_url:
        download_file(context, f"{base_url}/data.csv", "data.csv")

    change(tmp_path / "files")

    assert check_collected_data(context.deps) != []


def test_rerun_counts_the_files_of_an_earlier_run_against_the_limit(tmp_path):
    with serve_files({"/a.csv": b"a" * 100, "/big.csv": b"c" * 200}) as base_url:
        download_file(_context(tmp_path, max_bytes=250), f"{base_url}/a.csv", "a.csv")

        rerun = _context(tmp_path, max_bytes=250)
        over_limit = download_file(rerun, f"{base_url}/big.csv", "big.csv")

    assert "size limit" in over_limit
    assert current_budget(rerun.deps).used_bytes == 100


def test_rerun_can_discard_a_file_of_an_earlier_run(tmp_path):
    with serve_files({"/a.csv": b"a" * 100}) as base_url:
        download_file(_context(tmp_path), f"{base_url}/a.csv", "a.csv")

    discard_file(_context(tmp_path), "a.csv")

    assert _saved_files(tmp_path) == ["manifest.json"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == []


def test_discarding_a_path_that_was_not_downloaded_says_it_is_not_in_the_manifest(tmp_path):
    result = discard_file(_context(tmp_path), "missing.csv")

    assert "not in the manifest" in result


@pytest.mark.parametrize(
    ("filename", "archive", "unpacked_name", "unpacked_paths"),
    [
        ("data.zip", _zip({"a.csv": DATA, ".hidden.csv": DATA, "../escape.csv": DATA}), "data", ["data/a.csv", "data/.hidden.csv"]),
        ("data.tar.gz", _tar_gz({"a.csv": DATA, ".hidden.csv": DATA, "../escape.csv": DATA}), "data", ["data/a.csv", "data/.hidden.csv"]),
        ("data.csv.gz", gzip.compress(DATA), "data.csv", ["data.csv"]),
    ],
    ids=["zip", "tar.gz", "gz"],
)
def test_archive_is_unpacked_into_recorded_files(tmp_path, filename, archive, unpacked_name, unpacked_paths):
    context = _context(tmp_path)

    with serve_files({f"/{filename}": archive}) as base_url:
        download_file(context, f"{base_url}/{filename}", filename)

    entries = load_manifest(tmp_path)
    assert [entry.path for entry in entries] == unpacked_paths
    assert all(entry.archive == filename for entry in entries)
    assert (tmp_path / "files" / unpacked_paths[0]).read_bytes() == DATA
    assert check_collected_data(context.deps) == []

    discard_file(context, unpacked_name)

    assert _saved_files(tmp_path) == ["manifest.json"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == []


def test_archive_whose_unpacked_files_exceed_the_limit_is_not_kept(tmp_path):
    with serve_files({"/data.zip": _zip({"a.csv": b"x" * 1000, "b.csv": b"x" * 1000})}) as base_url:
        result = download_file(_context(tmp_path, max_bytes=1500), f"{base_url}/data.zip", "data.zip")

    assert result.startswith("Error")
    assert _saved_files(tmp_path) == []
