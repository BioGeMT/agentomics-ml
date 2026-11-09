import hashlib
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

FILES_DIRECTORY_NAME = "files"
MANIFEST_FILE_NAME = "manifest.json"
SUMMARY_FILE_NAME = "summary.md"
REVIEW_FILE_NAME = "review.json"


class ManifestEntry(BaseModel):
    path: str
    source_url: str
    size_bytes: int
    sha256: str
    archive: str | None = None


_MANIFEST = TypeAdapter(list[ManifestEntry])


def load_manifest(collected_data_directory: Path) -> list[ManifestEntry]:
    manifest_path = collected_data_directory / MANIFEST_FILE_NAME
    return _MANIFEST.validate_json(manifest_path.read_bytes()) if manifest_path.exists() else []


def save_manifest(collected_data_directory: Path, entries: list[ManifestEntry]) -> None:
    content = _MANIFEST.dump_json(entries, indent=2, exclude_none=True)
    write_atomically(collected_data_directory / MANIFEST_FILE_NAME, content)


def write_atomically(path: Path, content: bytes) -> None:
    """Replace the file at path in one step, so that an interrupted write leaves the earlier file."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def find_problems(collected_data_directory: Path, max_bytes: int | None, verify_hashes: bool) -> list[str]:
    """Return what makes the collected data folder disagree with its manifest or the size limit."""
    entries = load_manifest(collected_data_directory)
    problems = []
    for entry in entries:
        path = collected_data_directory / entry.path
        if not path.is_file() or path.stat().st_size != entry.size_bytes:
            problems.append(f"{entry.path} is missing or does not have the size the manifest records.")
        elif verify_hashes and _sha256(path) != entry.sha256:
            problems.append(f"{entry.path} does not have the SHA-256 the manifest records.")
    recorded_paths = {entry.path for entry in entries}
    for path in sorted((collected_data_directory / FILES_DIRECTORY_NAME).rglob("*")):
        relative_path = path.relative_to(collected_data_directory).as_posix()
        if path.is_file() and relative_path not in recorded_paths:
            problems.append(f"{relative_path} is not in the manifest.")
    used_bytes = sum(entry.size_bytes for entry in entries)
    if max_bytes is not None and used_bytes > max_bytes:
        problems.append(f"The files take {used_bytes} bytes, more than the size limit of {max_bytes} bytes.")
    return problems


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
