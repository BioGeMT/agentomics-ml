import gzip
import hashlib
import lzma
import tarfile
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import httpx2
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from agentomics.runtime.collected_data import FILES_DIRECTORY_NAME, ManifestEntry, load_manifest, save_manifest
from agentomics.utils.config import Config

SCRATCH_DIRECTORY_PREFIX = ".download-"
READABLE_ZIP_COMPRESSIONS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA)
# Bit 0 of a zip member's general purpose flags is set when the member is encrypted.
ZIP_ENCRYPTED_FLAG = 0x1
CHUNK_BYTES = 1 << 20
UNREADABLE_ARCHIVE_ERRORS = (
    zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, zlib.error, lzma.LZMAError, EOFError, UnicodeDecodeError,
)
MAX_LISTED_FILES = 20


class DownloadError(Exception):
    """A tool call that cannot be carried out; the message is for the agent."""


class Budget(BaseModel):
    """The size limit of the collected files and the bytes they take."""

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


class ToolResult(BaseModel):
    """What a download tool call did, as the agent receives it."""

    error: str | None = None
    content_type: str | None = None
    files: list[ManifestEntry] = []
    unlisted_file_count: int = 0
    budget: Budget


@dataclass
class _Download:
    url: str
    path: Path
    size_bytes: int
    sha256: str
    content_type: str | None


def create_download_tools() -> list[Tool[DownloadDeps]]:
    return [
        # Sequential: the tools share the manifest.
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


def _result(deps: DownloadDeps, files: Sequence[ManifestEntry] = (), **fields: str | None) -> ToolResult:
    return ToolResult(
        files=files[:MAX_LISTED_FILES],
        unlisted_file_count=max(len(files) - MAX_LISTED_FILES, 0),
        budget=current_budget(deps),
        **fields,
    )


def download_file(ctx: RunContext[DownloadDeps], url: str, filename: str) -> ToolResult:
    """Download a file from a public URL into the collected data folder and record it in the manifest.

    An archive is unpacked under the file name without its archive suffix, and the archive is removed.

    Args:
        url: The http or https URL of the file.
        filename: The name to save it under: a plain file name, without folders.
    """
    deps = ctx.deps
    try:
        target = _target_path(deps, url, filename)
        with tempfile.TemporaryDirectory(dir=deps.collected_data_directory, prefix=SCRATCH_DIRECTORY_PREFIX) as scratch:
            download = _fetch(deps, url, Path(scratch) / "downloaded")
            added = _store(deps, download, filename, target)
    except DownloadError as error:
        return _result(deps, error=str(error))
    save_manifest(deps.collected_data_directory, load_manifest(deps.collected_data_directory) + added)
    return _result(deps, files=added, content_type=download.content_type)


def _target_path(deps: DownloadDeps, url: str, filename: str) -> Path:
    """Return where the download goes, after checking that the tool call can be carried out."""
    suffix = _archive_suffix(filename)
    name = filename if suffix is None else filename.removesuffix(suffix)
    if Path(name).name != name or name in ("", ".."):
        raise DownloadError(f"'{filename}' is not a plain file name")
    target = deps.collected_data_directory / FILES_DIRECTORY_NAME / name
    if target.exists():
        raise DownloadError(f"{FILES_DIRECTORY_NAME}/{target.name} already exists")
    if not url.startswith(("http://", "https://")):
        raise DownloadError("only http and https URLs can be downloaded. Try the same address over https if the host offers it")
    return target


def _fetch(deps: DownloadDeps, url: str, path: Path) -> _Download:
    """Stream the URL to path."""
    budget = current_budget(deps)
    client = httpx2.Client(follow_redirects=True, timeout=deps.download_timeout)
    try:
        with client, client.stream("GET", url) as response:
            response.raise_for_status()
            if "content-length" in response.headers:
                # Refuse a file that cannot fit before downloading it.
                budget.model_copy().consume(int(response.headers["content-length"]))
            size_bytes, sha256 = _write_stream(response.iter_bytes(), path, budget)
    # UnicodeError: the host name cannot be encoded, for example because a label is empty.
    except (httpx2.HTTPError, httpx2.InvalidURL, UnicodeError) as error:
        raise DownloadError(f"the download failed: {error}") from error
    if size_bytes == 0:
        raise DownloadError("the download was empty")
    return _Download(url, path, size_bytes, sha256, response.headers.get("content-type"))


def _store(deps: DownloadDeps, download: _Download, filename: str, target: Path) -> list[ManifestEntry]:
    """Move the download, or the files unpacked from it, to target; return their manifest entries."""
    suffix = _archive_suffix(filename)
    if suffix is None:
        stored = download.path
        entries = [ManifestEntry(
            path=f"{FILES_DIRECTORY_NAME}/{target.name}",
            source_url=download.url,
            size_bytes=download.size_bytes,
            sha256=download.sha256,
        )]
    else:
        stored = download.path.with_name("unpacked") / target.name
        # The archive is removed, so only its unpacked files count.
        entries = _unpack(download, filename, suffix, stored, current_budget(deps))
    target.parent.mkdir(exist_ok=True)
    stored.rename(target)
    return entries


def _unpack(download: _Download, filename: str, suffix: str, destination: Path, budget: Budget) -> list[ManifestEntry]:
    """Unpack the files of the downloaded archive to destination; return their manifest entries."""
    # By path: a file that the archive holds twice is written twice and recorded once.
    entries = {}
    try:
        for member_name, source in _ARCHIVE_READERS[suffix](download.path):
            path = (PurePosixPath(destination.name) / member_name).as_posix()
            size_bytes, sha256 = _write_stream(iter(lambda: source.read(CHUNK_BYTES), b""), destination.parent / path, budget)
            entries[path] = ManifestEntry(
                path=f"{FILES_DIRECTORY_NAME}/{path}",
                source_url=download.url,
                size_bytes=size_bytes,
                sha256=sha256,
                archive=filename,
            )
    except UNREADABLE_ARCHIVE_ERRORS as error:
        raise DownloadError(f"the archive could not be unpacked: {error}") from error
    if not entries:
        raise DownloadError("the archive holds no files")
    return list(entries.values())


def _write_stream(chunks: Iterable[bytes], path: Path, budget: Budget) -> tuple[int, str]:
    """Write the chunks to path and count them against the budget; return the size and the SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("wb") as output:
        for chunk in chunks:
            budget.consume(len(chunk))
            size_bytes += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    return size_bytes, digest.hexdigest()


def _zip_members(archive: Path) -> Iterator[tuple[str, BinaryIO]]:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not member.is_dir() and _is_plain_relative_path(member.filename):
                if member.flag_bits & ZIP_ENCRYPTED_FLAG or member.compress_type not in READABLE_ZIP_COMPRESSIONS:
                    raise DownloadError(f"{member.filename} in the archive is encrypted or uses an unsupported compression")
                with bundle.open(member) as source:
                    yield member.filename, source


def _tar_members(archive: Path) -> Iterator[tuple[str, BinaryIO]]:
    # Strict: a file name that is not UTF-8 is an error, because the manifest cannot record it.
    with tarfile.open(archive, errors="strict") as bundle:
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
ARCHIVE_SUFFIXES = tuple(_ARCHIVE_READERS)


def _archive_suffix(filename: str) -> str | None:
    """Return the longest archive suffix the file name ends with, unless the name is only the suffix."""
    suffixes = [suffix for suffix in _ARCHIVE_READERS if filename.endswith(suffix) and filename != suffix]
    return max(suffixes, key=len, default=None)


def discard_file(ctx: RunContext[DownloadDeps], path: str) -> ToolResult:
    """Delete a downloaded file that does not fit the request, and remove it from the manifest.

    Args:
        path: The path of the file as the manifest records it, or of the folder an archive was unpacked into.
    """
    deps = ctx.deps
    path = path.rstrip("/")
    entries = load_manifest(deps.collected_data_directory)
    discarded = [entry for entry in entries if entry.path == path or entry.path.startswith(f"{path}/")]
    if not discarded:
        return _result(deps, error=f"{path} is not in the manifest")
    for entry in discarded:
        file_path = deps.collected_data_directory / entry.path
        file_path.unlink()
        _remove_empty_folders(file_path.parent, deps.collected_data_directory / FILES_DIRECTORY_NAME)
    save_manifest(deps.collected_data_directory, [entry for entry in entries if entry not in discarded])
    return _result(deps, files=discarded)


def _remove_empty_folders(folder: Path, stop: Path) -> None:
    """Remove folder and its parents below stop while they are empty."""
    while folder != stop and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def _is_plain_relative_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and not PurePosixPath(path).is_absolute() and ".." not in parts
