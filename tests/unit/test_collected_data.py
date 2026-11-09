import hashlib
from pathlib import Path

import pytest

from agentomics.runtime.collected_data import ManifestEntry, find_problems, save_manifest

DATA = b"x" * 100


def _write_recorded_file(collected_data_directory: Path) -> Path:
    """Create a collected data folder with one file that its manifest records."""
    path = collected_data_directory / "files" / "data.csv"
    path.parent.mkdir()
    path.write_bytes(DATA)
    entry = ManifestEntry(
        path="files/data.csv",
        source_url="https://example.org/data.csv",
        size_bytes=len(DATA),
        sha256=hashlib.sha256(DATA).hexdigest(),
    )
    save_manifest(collected_data_directory, [entry])
    return path


def test_folder_that_matches_its_manifest_has_no_problems(tmp_path):
    _write_recorded_file(tmp_path)

    assert find_problems(tmp_path, max_bytes=len(DATA), verify_hashes=True) == []


@pytest.mark.parametrize(("verify_hashes", "problem_count"), [(True, 1), (False, 0)])
def test_changed_bytes_of_the_same_size_are_found_only_when_hashes_are_verified(tmp_path, verify_hashes, problem_count):
    _write_recorded_file(tmp_path).write_bytes(b"y" * len(DATA))

    problems = find_problems(tmp_path, max_bytes=None, verify_hashes=verify_hashes)

    assert len(problems) == problem_count
    assert all("SHA-256" in problem for problem in problems)


def test_recorded_file_that_is_missing_is_a_problem(tmp_path):
    _write_recorded_file(tmp_path).unlink()

    problems = find_problems(tmp_path, max_bytes=None, verify_hashes=False)

    assert len(problems) == 1
    assert "missing" in problems[0]


def test_file_that_the_manifest_does_not_record_is_a_problem(tmp_path):
    _write_recorded_file(tmp_path).with_name("extra.csv").write_bytes(b"extra")

    assert find_problems(tmp_path, max_bytes=None, verify_hashes=False) == ["files/extra.csv is not in the manifest."]


def test_files_that_take_more_than_the_size_limit_are_a_problem(tmp_path):
    _write_recorded_file(tmp_path)

    problems = find_problems(tmp_path, max_bytes=len(DATA) - 1, verify_hashes=False)

    assert len(problems) == 1
    assert "size limit" in problems[0]
