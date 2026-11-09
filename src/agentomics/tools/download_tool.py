import gzip
import hashlib
import lzma
import tarfile
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import httpx2
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from agentomics.runtime.collected_data import ManifestEntry, load_manifest, save_manifest
from agentomics.utils.config import Config

FILES_DIRECTORY_NAME = "files"
SCRATCH_DIRECTORY_PREFIX = ".download-"
READABLE_ZIP_COMPRESSIONS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA)
# Bit 0 of a zip member's general purpose flags is set when the member is encrypted.
ZIP_ENCRYPTED_FLAG = 0x1
CHUNK_BYTES = 1 << 20
UNREADABLE_ARCHIVE_ERRORS = (zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, zlib.error, lzma.LZMAError, EOFError)
LISTED_UNPACKED_FILES = 20


class DownloadError(Exception):
    """A download that cannot be kept; the message is for the agent."""


class Budget(BaseModel):
    """The size limit of the downloaded files and the bytes they take."""

    max_bytes: int | None
    used_bytes: int

    def consume(self, size_bytes: int) -> None:
        if self.max_bytes is not None and self.used_bytes + size_bytes > self.max_bytes:
            raise DownloadError(
                f"the size limit of {self.max_bytes} bytes would be exceeded ({self.used_bytes} bytes are used)"
            )
        self.used_bytes += size_bytes


@dataclass(kw_only=True)
class DownloadDeps:
    """What the download tools need from the run that uses them."""

    collected_data_directory: Path
    max_bytes: int | None
    download_timeout: int = Config.DEFAULT_DOWNLOAD_TOOL_TIMEOUT


def create_download_tools() -> list[Tool[DownloadDeps]]:
    return [
        # Sequential: the tools share the manifest and the partial-download file.
        Tool(
            function,
            takes_ctx=True,
            max_retries=Config.DEFAULT_MAX_TOOL_RETRIES,
            require_parameter_descriptions=True,
            sequential=True,
        )
        for function in (download_file, discard_file)
    ]


def current_budget(deps: DownloadDeps) -> Budget:
    used_bytes = sum(entry.size_bytes for entry in load_manifest(deps.collected_data_directory))
    return Budget(max_bytes=deps.max_bytes, used_bytes=used_bytes)


def budget_text(deps: DownloadDeps) -> str:
    budget = current_budget(deps)
    if budget.max_bytes is None:
        return f"The folder holds {budget.used_bytes} bytes; there is no size limit."
    return f"The folder holds {budget.used_bytes} bytes; the size limit is {budget.max_bytes} bytes."


def download_file(ctx: RunContext[DownloadDeps], url: str, filename: str) -> str:
    """Download a file from a public URL into the collected data folder and record it in the manifest.

    Zip, tar and gz archives are unpacked into files/ under the file name without its archive suffix.

    Args:
        url: The http or https URL of the file.
        filename: The name to save it under: a plain file name, without folders.
    """
    deps = ctx.deps
    try:
        target = _target_path(deps, url, filename)
        with tempfile.TemporaryDirectory(dir=deps.collected_data_directory, prefix=SCRATCH_DIRECTORY_PREFIX) as scratch:
            downloaded = Path(scratch) / "downloaded"
            size, sha256, content_type = _fetch(url, downloaded, current_budget(deps), deps.download_timeout)
            added = _store(deps, url, filename, downloaded, size, sha256, target)
    except DownloadError as error:
        return f"Error: {error}. {budget_text(deps)}"

    save_manifest(deps.collected_data_directory, load_manifest(deps.collected_data_directory) + added)
    if _archive_suffix(filename) is None:
        return f"Saved {added[0].path} ({size} bytes, {content_type}). {budget_text(deps)}"
    listing = "\n".join(entry.path for entry in added[:LISTED_UNPACKED_FILES])
    if len(added) > LISTED_UNPACKED_FILES:
        listing += f"\n... and {len(added) - LISTED_UNPACKED_FILES} more"
    unpacked_bytes = sum(entry.size_bytes for entry in added)
    return (f"Downloaded {filename} ({size} bytes, {content_type}), unpacked it into {FILES_DIRECTORY_NAME}/{target.name} and "
            f"removed the archive: {len(added)} files, {unpacked_bytes} bytes:\n{listing}\n{budget_text(deps)}")


def _target_path(deps: DownloadDeps, url: str, filename: str) -> Path:
    """Return where the download goes in files/, after checking that the request can be carried out."""
    if Path(filename).name != filename or filename in ("", ".."):
        raise DownloadError(f"'{filename}' is not a plain file name")
    suffix = _archive_suffix(filename)
    target = deps.collected_data_directory / FILES_DIRECTORY_NAME / (filename if suffix is None else filename.removesuffix(suffix))
    if target.exists() or target.is_symlink():
        raise DownloadError(f"files/{target.name} already exists")
    if not url.startswith(("http://", "https://")):
        raise DownloadError("only http and https URLs can be downloaded. Try the same address over https if the host offers it")
    return target


def _fetch(url: str, path: Path, budget: Budget, timeout: int) -> tuple[int, str, str]:
    """Stream the URL to path; return the size, the SHA-256 and the content type."""
    try:
        with httpx2.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            if "content-length" in response.headers:
                # Refuse a file that cannot fit before downloading it.
                budget.model_copy().consume(int(response.headers["content-length"]))
            size, sha256 = _write_stream(response.iter_bytes(), path, budget)
    except (httpx2.HTTPError, httpx2.InvalidURL) as error:
        raise DownloadError(f"the download failed: {error}") from error
    if size == 0:
        raise DownloadError("the download was empty")
    return size, sha256, response.headers.get("content-type", "unknown type")


def _store(
    deps: DownloadDeps, url: str, filename: str, downloaded: Path, size: int, sha256: str, target: Path,
) -> list[ManifestEntry]:
    """Move the download, or the files unpacked from it, to target; return their manifest entries."""
    suffix = _archive_suffix(filename)
    if suffix is None:
        added = [ManifestEntry(path=f"{FILES_DIRECTORY_NAME}/{target.name}", source_url=url, size_bytes=size, sha256=sha256)]
    else:
        unpacked_root = downloaded.with_name("unpacked")
        # The archive is removed, so only its unpacked files count.
        unpacked = _unpack(downloaded, suffix, target.name, unpacked_root, current_budget(deps))
        added = [
            ManifestEntry(
                path=f"{FILES_DIRECTORY_NAME}/{path}", source_url=url, size_bytes=member_size, sha256=member_sha256,
                archive=filename,
            )
            for path, (member_size, member_sha256) in unpacked.items()
        ]
        downloaded = unpacked_root / target.name
    target.parent.mkdir(exist_ok=True)
    downloaded.rename(target)
    return added


def _unpack(archive: Path, suffix: str, name: str, unpacked_root: Path, budget: Budget) -> dict[str, tuple[int, str]]:
    """Unpack the archive's files under unpacked_root/name; return the size and the SHA-256 of each by its path."""
    unpacked = {}
    try:
        for member_name, source in _ARCHIVE_READERS[suffix](archive):
            path = (PurePosixPath(name) / member_name).as_posix()
            unpacked[path] = _write_stream(iter(lambda: source.read(CHUNK_BYTES), b""), unpacked_root / path, budget)
    except UNREADABLE_ARCHIVE_ERRORS as error:
        raise DownloadError(f"the archive could not be unpacked: {error}") from error
    if not unpacked:
        raise DownloadError("the archive holds no files")
    return unpacked


def _write_stream(chunks: Iterable[bytes], path: Path, budget: Budget) -> tuple[int, str]:
    """Write the chunks to path and count them against the budget; return the size and the SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as output:
        for chunk in chunks:
            budget.consume(len(chunk))
            size += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    return size, digest.hexdigest()


def _zip_members(archive: Path) -> Iterator[tuple[str, BinaryIO]]:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not member.is_dir() and _is_plain_relative_path(member.filename):
                if member.flag_bits & ZIP_ENCRYPTED_FLAG or member.compress_type not in READABLE_ZIP_COMPRESSIONS:
                    raise DownloadError(f"{member.filename} in the archive is encrypted or uses an unsupported compression")
                with bundle.open(member) as source:
                    yield member.filename, source


def _tar_members(archive: Path) -> Iterator[tuple[str, BinaryIO]]:
    with tarfile.open(archive) as bundle:
        for member in bundle:
            if member.isfile() and _is_plain_relative_path(member.name):
                with bundle.extractfile(member) as source:
                    yield member.name, source


def _gzip_member(archive: Path) -> Iterator[tuple[str, BinaryIO]]:
    with gzip.open(archive) as source:
        yield "", source


# How the files of an archive are read, by the suffix of its file name. A gz file holds one file without a name.
_ARCHIVE_READERS: dict[str, Callable[[Path], Iterator[tuple[str, BinaryIO]]]] = {
    ".zip": _zip_members,
    ".tar": _tar_members,
    ".tar.gz": _tar_members,
    ".tgz": _tar_members,
    ".tar.bz2": _tar_members,
    ".tbz2": _tar_members,
    ".tar.xz": _tar_members,
    ".txz": _tar_members,
    ".gz": _gzip_member,
}


def _archive_suffix(filename: str) -> str | None:
    """Return the longest archive suffix the file name ends with, unless the name is only the suffix."""
    suffixes = [suffix for suffix in _ARCHIVE_READERS if filename.endswith(suffix) and filename != suffix]
    return max(suffixes, key=len, default=None)


def discard_file(ctx: RunContext[DownloadDeps], path: str) -> str:
    """Delete a downloaded file that does not fit the request, and remove it from the manifest.

    Args:
        path: The path of the file as the manifest records it, or of the folder an archive was unpacked into.
    """
    deps = ctx.deps
    path = path.rstrip("/")
    entries = load_manifest(deps.collected_data_directory)
    discarded = [entry for entry in entries if entry.path == path or entry.path.startswith(f"{path}/")]
    if not discarded:
        return f"Error: {path} is not in the manifest. {budget_text(deps)}"
    for entry in discarded:
        file_path = deps.collected_data_directory / entry.path
        file_path.unlink()
        _remove_empty_folders(file_path.parent, deps.collected_data_directory / FILES_DIRECTORY_NAME)
    save_manifest(deps.collected_data_directory, [entry for entry in entries if entry not in discarded])
    return f"Discarded {path}. {budget_text(deps)}"


def _remove_empty_folders(folder: Path, stop: Path) -> None:
    """Remove folder and its parents below stop while they are empty."""
    while folder != stop and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def check_collected_data(deps: DownloadDeps, verify_hashes: bool) -> list[str]:
    """Return the problems that make the collected data folder disagree with its manifest or the size limit."""
    files_directory = deps.collected_data_directory / FILES_DIRECTORY_NAME
    if files_directory.is_symlink():
        return ["files/ is a symbolic link."]
    entries = load_manifest(deps.collected_data_directory)
    problems = []
    recorded = [entry.path for entry in entries]
    if len(recorded) != len(set(recorded)):
        problems.append("The manifest records a file more than once.")
    for entry in entries:
        path = deps.collected_data_directory / entry.path
        if not _is_plain_relative_path(entry.path):
            problems.append(f"The manifest path '{entry.path}' is not a plain path inside files/.")
        elif path.is_symlink() or not path.is_file():
            problems.append(f"{entry.path} is in the manifest but is missing or not a regular file.")
        elif path.stat().st_size == 0:
            problems.append(f"{entry.path} is empty.")
        elif path.stat().st_size != entry.size_bytes or (verify_hashes and _sha256(path) != entry.sha256):
            problems.append(f"{entry.path} differs from its manifest entry.")
    for path in sorted(files_directory.rglob("*")):
        relative = path.relative_to(deps.collected_data_directory).as_posix()
        if path.is_symlink():
            problems.append(f"{relative} is a symbolic link.")
        elif path.is_file() and relative not in recorded:
            problems.append(f"{relative} is not in the manifest.")
    budget = current_budget(deps)
    if budget.max_bytes is not None and budget.used_bytes > budget.max_bytes:
        problems.append(f"The folder holds {budget.used_bytes} bytes, more than the size limit of {budget.max_bytes} bytes.")
    return problems


def _is_plain_relative_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and not PurePosixPath(path).is_absolute() and ".." not in parts


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
