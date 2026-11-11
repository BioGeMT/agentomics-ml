from __future__ import annotations

from pathlib import Path

import libarchive

from agentomics.runtime.filesystem import ByteBudget, write_chunks

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".gz")


def get_archive_suffix(filename: str) -> str | None:
    suffixes = [suffix for suffix in ARCHIVE_SUFFIXES if filename.endswith(suffix) and filename != suffix]
    return max(suffixes, key=len, default=None)


def unpack_archive(archive_path: Path, destination: Path, byte_budget: ByteBudget) -> None:
    # A gz file that is not a tar holds one file without a name. libarchive reads it only as "raw",
    # and it is unpacked to destination itself.
    is_single_file = get_archive_suffix(archive_path.name) == ".gz"
    try:
        with libarchive.file_reader(str(archive_path), format_name="raw" if is_single_file else "all") as archive:
            for entry in archive:
                if not entry.isfile:
                    continue
                if isinstance(entry.pathname, bytes):
                    raise ValueError("a file name in the archive is not UTF-8")
                path = destination if is_single_file else destination / entry.pathname
                if not path.resolve().is_relative_to(destination.resolve()):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                write_chunks(entry.get_blocks(), path, byte_budget)
    except libarchive.ArchiveError as error:
        raise ValueError(f"the archive could not be unpacked: {error}") from error
    if not destination.exists():
        raise ValueError("the archive holds no files")
