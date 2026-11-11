import io
import tarfile

import pytest

from agentomics.runtime.archives import get_archive_suffix, unpack_archive
from agentomics.runtime.filesystem import ByteBudget
from tests.helpers import create_zip_archive, list_files


@pytest.mark.parametrize(
    ("filename", "suffix"),
    [("data.tar.gz", ".tar.gz"), ("data.csv.gz", ".gz"), ("data.csv", None), (".zip", None)],
)
def test_archive_suffix_is_the_longest_one_the_file_name_ends_with(filename, suffix):
    assert get_archive_suffix(filename) == suffix


def test_entry_whose_path_leaves_the_destination_is_not_unpacked(tmp_path):
    archive_path = tmp_path / "data.zip"
    archive_path.write_bytes(create_zip_archive([("a.csv", b"a"), ("../escape.csv", b"b"), ("/absolute.csv", b"c")]))

    unpack_archive(archive_path, tmp_path / "unpacked", ByteBudget(max_bytes=None, used_bytes=0))

    assert list_files(tmp_path / "unpacked") == ["a.csv"]
    assert not (tmp_path / "escape.csv").exists()


def test_archive_with_a_file_name_that_is_not_utf_8_cannot_be_unpacked(tmp_path):
    archive_path = tmp_path / "data.tar.gz"
    member = tarfile.TarInfo(b"caf\xe9.csv".decode(errors="surrogateescape"))
    member.size = 1
    with tarfile.open(archive_path, mode="w:gz") as bundle:
        bundle.addfile(member, io.BytesIO(b"a"))

    with pytest.raises(ValueError, match="not UTF-8"):
        unpack_archive(archive_path, tmp_path / "unpacked", ByteBudget(max_bytes=None, used_bytes=0))
