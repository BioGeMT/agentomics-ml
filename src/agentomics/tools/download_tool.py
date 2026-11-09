import gzip
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import httpx2

BYTES_PER_GB = 1_000_000_000
FILES_DIRECTORY_NAME = "files"
MANIFEST_FILE_NAME = "manifest.json"
PARTIAL_FILE_NAME = ".download.part"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip", ".gz")
READABLE_ZIP_COMPRESSIONS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA)
# Bit 0 of a zip member's general purpose flags is set when the member is encrypted.
ZIP_ENCRYPTED_FLAG = 0x1
CHUNK_BYTES = 1 << 20
LISTED_UNPACKED_FILES = 20


class DownloadBudgetExceeded(Exception):
    pass


class DownloadFolder:
    """The output folder of the data collection agent: downloaded files and the manifest that records them."""

    def __init__(self, root: Path, max_download_gb: float | None):
        self.root = root
        self.files_directory = root / FILES_DIRECTORY_NAME
        self.manifest_path = root / MANIFEST_FILE_NAME
        self.max_bytes = None if max_download_gb is None else int(max_download_gb * BYTES_PER_GB)
        self.files_directory.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else []
        )

    def retained_bytes(self) -> int:
        return sum(entry["size_bytes"] for entry in self.entries)

    def remaining_bytes(self) -> int | None:
        return None if self.max_bytes is None else self.max_bytes - self.retained_bytes()

    def budget_text(self) -> str:
        remaining = self.remaining_bytes()
        if remaining is None:
            return f"The folder holds {self.retained_bytes()} bytes; there is no size limit."
        return f"The folder holds {self.retained_bytes()} bytes; {max(remaining, 0)} bytes of the limit remain."

    def download_file(self, url: str, filename: str) -> str:
        """Download a file from a public URL into the output folder and record it in the manifest.

        Zip, tar and gz archives are unpacked into files/ under the file name without its archive suffix.

        Args:
            url: The http or https URL of the file.
            filename: The name to save it under: a plain file name, without folders.
        """
        if Path(filename).name != filename or filename in ("", ".."):
            return f"Error: '{filename}' is not a plain file name. {self.budget_text()}"
        unpacked_name = _unpacked_name(filename)
        target = self.files_directory / (unpacked_name or filename)
        if target.exists() or target.is_symlink():
            return f"Error: files/{target.name} already exists. {self.budget_text()}"
        if not url.startswith(("http://", "https://")):
            return (f"Error: only http and https URLs can be downloaded. Try the same address over https if the host "
                    f"offers it. {self.budget_text()}")

        remaining = self.remaining_bytes()
        partial = self.root / PARTIAL_FILE_NAME
        digest = hashlib.sha256()
        size = 0
        try:
            with httpx2.stream("GET", url, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "unknown type")
                advertised = response.headers.get("content-length")
                if remaining is not None and advertised is not None and int(advertised) > remaining:
                    raise DownloadBudgetExceeded(f"the file has {advertised} bytes, more than the limit allows")
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if remaining is not None and size > remaining:
                            raise DownloadBudgetExceeded(f"the file has more than the {remaining} bytes the limit allows")
                        digest.update(chunk)
                        output.write(chunk)
            if size == 0:
                return f"Error: the download was empty. {self.budget_text()}"
            if unpacked_name is None:
                partial.rename(target)
                added = [{"path": filename, "source_url": url, "size_bytes": size, "sha256": digest.hexdigest()}]
            else:
                added = self._unpack(partial, filename, target, url, remaining)
        except DownloadBudgetExceeded as error:
            return f"Error: not kept: {error}. {self.budget_text()}"
        except (zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, zlib.error, EOFError) as error:
            return f"Error: {filename} could not be unpacked: {error}. {self.budget_text()}"
        except (httpx2.HTTPError, OSError) as error:
            return f"Error: download failed: {error}. {self.budget_text()}"
        finally:
            partial.unlink(missing_ok=True)

        self.entries.extend(added)
        self._write_manifest()
        if unpacked_name is None:
            return f"Saved files/{filename} ({size} bytes, {content_type}). {self.budget_text()}"
        listing = "\n".join(f"files/{entry['path']}" for entry in added[:LISTED_UNPACKED_FILES])
        if len(added) > LISTED_UNPACKED_FILES:
            listing += f"\n... and {len(added) - LISTED_UNPACKED_FILES} more"
        unpacked_bytes = sum(entry["size_bytes"] for entry in added)
        return (f"Downloaded {filename} ({size} bytes, {content_type}), unpacked it into files/{unpacked_name} and "
                f"removed the archive: {len(added)} files, {unpacked_bytes} bytes:\n{listing}\n{self.budget_text()}")

    def discard_file(self, path: str) -> str:
        """Delete a downloaded file that does not fit the request, and remove it from the manifest.

        Args:
            path: The path in files/: a file, or the folder an archive was unpacked into.
        """
        discarded = [entry for entry in self.entries if entry["path"] == path or entry["path"].startswith(f"{path}/")]
        if not discarded:
            return f"Error: only files downloaded in this run can be discarded. {self.budget_text()}"
        target = self.files_directory / path
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        self.entries = [entry for entry in self.entries if entry not in discarded]
        self._write_manifest()
        return f"Discarded files/{path}. {self.budget_text()}"

    def check(self) -> list[str]:
        """Return the problems that make the folder disagree with its manifest or the size limit."""
        if self.files_directory.is_symlink():
            return ["files/ is a symbolic link."]
        problems = []
        recorded = [entry["path"] for entry in self.entries]
        if len(recorded) != len(set(recorded)):
            problems.append("The manifest records a file more than once.")
        for entry in self.entries:
            path = self.files_directory / entry["path"]
            if not _is_plain_relative_path(entry["path"]):
                problems.append(f"The manifest path '{entry['path']}' is not a plain path inside files/.")
            elif path.is_symlink() or not path.is_file():
                problems.append(f"files/{entry['path']} is in the manifest but is missing or not a regular file.")
            elif path.stat().st_size == 0:
                problems.append(f"files/{entry['path']} is empty.")
            elif path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
                problems.append(f"files/{entry['path']} differs from its manifest entry.")
        for path in sorted(self.files_directory.rglob("*")):
            relative = path.relative_to(self.files_directory).as_posix()
            if path.is_symlink():
                problems.append(f"files/{relative} is a symbolic link.")
            elif path.is_file() and relative not in recorded:
                problems.append(f"files/{relative} is not in the manifest.")
        if self.max_bytes is not None and self.retained_bytes() > self.max_bytes:
            problems.append(f"The folder holds {self.retained_bytes()} bytes, more than the limit of {self.max_bytes}.")
        return problems

    def _unpack(self, archive: Path, filename: str, target: Path, url: str, remaining: int | None) -> list[dict]:
        added: list[dict] = []

        def save(source: BinaryIO, relative: str) -> None:
            left = None if remaining is None else remaining - sum(entry["size_bytes"] for entry in added)
            entry = self._save_unpacked(source, relative, left)
            if entry is not None:
                added.append({**entry, "source_url": url, "archive": filename})

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

    def _save_unpacked(self, source: BinaryIO, relative: str, remaining: int | None) -> dict | None:
        relative = PurePosixPath(relative).as_posix()
        path = self.files_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with path.open("xb") as output:
            for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                size += len(chunk)
                if remaining is not None and size > remaining:
                    raise DownloadBudgetExceeded(f"its unpacked files need more than the {remaining} bytes the limit allows")
                digest.update(chunk)
                output.write(chunk)
        if size == 0:
            path.unlink()
            return None
        return {"path": relative, "size_bytes": size, "sha256": digest.hexdigest()}

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(self.entries, indent=2))
        os.replace(temporary, self.manifest_path)


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
