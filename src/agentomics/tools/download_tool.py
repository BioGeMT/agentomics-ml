import gzip
import hashlib
import shutil
import tarfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import httpx2
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from agentomics.runtime.collected_data import ManifestEntry, load_manifest, save_manifest
from agentomics.utils.config import Config

FILES_DIRECTORY_NAME = "files"
PARTIAL_FILE_NAME = ".download.part"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip", ".gz")
READABLE_ZIP_COMPRESSIONS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA)
# Bit 0 of a zip member's general purpose flags is set when the member is encrypted.
ZIP_ENCRYPTED_FLAG = 0x1
CHUNK_BYTES = 1 << 20
LISTED_UNPACKED_FILES = 20


class DownloadBudgetExceeded(Exception):
    pass


class Budget(BaseModel):
    """The size limit of the downloaded files and the bytes they take."""

    max_bytes: int | None
    used_bytes: int

    def consume(self, size_bytes: int) -> None:
        if self.max_bytes is not None and self.used_bytes + size_bytes > self.max_bytes:
            raise DownloadBudgetExceeded(
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
    files_directory = deps.collected_data_directory / FILES_DIRECTORY_NAME
    if Path(filename).name != filename or filename in ("", ".."):
        return f"Error: '{filename}' is not a plain file name. {budget_text(deps)}"
    unpacked_name = _unpacked_name(filename)
    target = files_directory / (unpacked_name or filename)
    if target.exists() or target.is_symlink():
        return f"Error: files/{target.name} already exists. {budget_text(deps)}"
    if not url.startswith(("http://", "https://")):
        return (f"Error: only http and https URLs can be downloaded. Try the same address over https if the host "
                f"offers it. {budget_text(deps)}")

    budget = current_budget(deps)
    partial = deps.collected_data_directory / PARTIAL_FILE_NAME
    digest = hashlib.sha256()
    size = 0
    try:
        with httpx2.stream("GET", url, follow_redirects=True, timeout=deps.download_timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "unknown type")
            if "content-length" in response.headers:
                # Refuse a file that cannot fit before downloading it.
                budget.model_copy().consume(int(response.headers["content-length"]))
            with partial.open("wb") as output:
                for chunk in response.iter_bytes():
                    budget.consume(len(chunk))
                    size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
        if size == 0:
            return f"Error: the download was empty. {budget_text(deps)}"
        files_directory.mkdir(exist_ok=True)
        if unpacked_name is None:
            partial.rename(target)
            added = [ManifestEntry(path=filename, source_url=url, size_bytes=size, sha256=digest.hexdigest())]
        else:
            added = _unpack(deps, partial, filename, target, url)
    except DownloadBudgetExceeded as error:
        return f"Error: not kept: {error}. {budget_text(deps)}"
    except (zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, zlib.error, EOFError) as error:
        return f"Error: {filename} could not be unpacked: {error}. {budget_text(deps)}"
    except (httpx2.HTTPError, OSError) as error:
        return f"Error: download failed: {error}. {budget_text(deps)}"
    finally:
        partial.unlink(missing_ok=True)

    save_manifest(deps.collected_data_directory, load_manifest(deps.collected_data_directory) + added)
    if unpacked_name is None:
        return f"Saved files/{filename} ({size} bytes, {content_type}). {budget_text(deps)}"
    listing = "\n".join(f"files/{entry.path}" for entry in added[:LISTED_UNPACKED_FILES])
    if len(added) > LISTED_UNPACKED_FILES:
        listing += f"\n... and {len(added) - LISTED_UNPACKED_FILES} more"
    unpacked_bytes = sum(entry.size_bytes for entry in added)
    return (f"Downloaded {filename} ({size} bytes, {content_type}), unpacked it into files/{unpacked_name} and "
            f"removed the archive: {len(added)} files, {unpacked_bytes} bytes:\n{listing}\n{budget_text(deps)}")


def discard_file(ctx: RunContext[DownloadDeps], path: str) -> str:
    """Delete a downloaded file that does not fit the request, and remove it from the manifest.

    Args:
        path: The path in files/: a file, or the folder an archive was unpacked into.
    """
    deps = ctx.deps
    entries = load_manifest(deps.collected_data_directory)
    discarded = [entry for entry in entries if entry.path == path or entry.path.startswith(f"{path}/")]
    if not discarded:
        return f"Error: files/{path} is not in the manifest. {budget_text(deps)}"
    target = deps.collected_data_directory / FILES_DIRECTORY_NAME / path
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    save_manifest(deps.collected_data_directory, [entry for entry in entries if entry not in discarded])
    return f"Discarded files/{path}. {budget_text(deps)}"


def check_collected_data(deps: DownloadDeps) -> list[str]:
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
        path = files_directory / entry.path
        if not _is_plain_relative_path(entry.path):
            problems.append(f"The manifest path '{entry.path}' is not a plain path inside files/.")
        elif path.is_symlink() or not path.is_file():
            problems.append(f"files/{entry.path} is in the manifest but is missing or not a regular file.")
        elif path.stat().st_size == 0:
            problems.append(f"files/{entry.path} is empty.")
        elif path.stat().st_size != entry.size_bytes or _sha256(path) != entry.sha256:
            problems.append(f"files/{entry.path} differs from its manifest entry.")
    for path in sorted(files_directory.rglob("*")):
        relative = path.relative_to(files_directory).as_posix()
        if path.is_symlink():
            problems.append(f"files/{relative} is a symbolic link.")
        elif path.is_file() and relative not in recorded:
            problems.append(f"files/{relative} is not in the manifest.")
    budget = current_budget(deps)
    if budget.max_bytes is not None and budget.used_bytes > budget.max_bytes:
        problems.append(f"The folder holds {budget.used_bytes} bytes, more than the size limit of {budget.max_bytes} bytes.")
    return problems


def _unpack(deps: DownloadDeps, archive: Path, filename: str, target: Path, url: str) -> list[ManifestEntry]:
    added: list[ManifestEntry] = []
    # The archive is removed, so only its unpacked files count.
    budget = current_budget(deps)

    def save(source: BinaryIO, relative: str) -> None:
        relative = PurePosixPath(relative).as_posix()
        saved = _save_unpacked(source, target.parent / relative, budget)
        if saved is not None:
            size, sha256 = saved
            added.append(ManifestEntry(path=relative, source_url=url, size_bytes=size, sha256=sha256, archive=filename))

    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if not member.is_dir() and _is_plain_relative_path(member.filename):
                        if member.flag_bits & ZIP_ENCRYPTED_FLAG or member.compress_type not in READABLE_ZIP_COMPRESSIONS:
                            raise zipfile.BadZipFile(f"{member.filename} is encrypted or uses an unsupported compression")
                        with bundle.open(member) as source:
                            save(source, f"{target.name}/{member.filename}")
        elif filename.endswith(".gz") and not filename.endswith(".tar.gz"):
            with gzip.open(archive) as source:
                save(source, target.name)
        else:
            with tarfile.open(archive) as bundle:
                for member in bundle:
                    if member.isfile() and _is_plain_relative_path(member.name):
                        with bundle.extractfile(member) as source:
                            save(source, f"{target.name}/{member.name}")
        if not added:
            raise EOFError("it holds no files")
    except BaseException:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        raise
    return added


def _save_unpacked(source: BinaryIO, path: Path, budget: Budget) -> tuple[int, str] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as output:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            budget.consume(len(chunk))
            size += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    if size == 0:
        path.unlink()
        return None
    return size, digest.hexdigest()


def _unpacked_name(filename: str) -> str | None:
    for suffix in ARCHIVE_SUFFIXES:
        if filename.endswith(suffix) and len(filename) > len(suffix):
            return filename[: -len(suffix)]
    return None


def _is_plain_relative_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and not PurePosixPath(path).is_absolute() and ".." not in parts


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
