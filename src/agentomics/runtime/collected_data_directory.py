from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from agentomics.runtime.filesystem import get_file_hash, remove_empty_directories, remove_path, write_atomically
from agentomics.utils.data_module_config import DataModuleConfig


class ManifestEntry(BaseModel):
    path: str
    # The URL that was downloaded. A file unpacked from an archive has the URL of the archive.
    source_url: str
    size_bytes: int
    sha256: str
    archive: str | None = None


_MANIFEST = TypeAdapter(list[ManifestEntry])


def initialize_collected_data_directory(config: DataModuleConfig) -> None:
    config.collected_data_dir.mkdir(parents=True, exist_ok=True)
    if not config.manifest_path.exists():
        _save_manifest(config, [])


def validate_collected_data_directory(config: DataModuleConfig) -> None:
    entries = load_manifest(config)
    problems = []
    for entry in entries:
        path = config.collected_data_dir / entry.path
        if not path.is_file() or path.stat().st_size != entry.size_bytes:
            problems.append(f"{entry.path} is missing or does not have the size the manifest records.")
    recorded_paths = {entry.path for entry in entries}
    for path in sorted(config.files_dir.rglob("*")):
        if path.is_file() and _get_manifest_path(config, path) not in recorded_paths:
            problems.append(f"{_get_manifest_path(config, path)} is not in the manifest.")
    if problems:
        raise ValueError(f"{config.collected_data_dir} does not match its manifest:\n" + "\n".join(problems))


def read_manifest(config: DataModuleConfig) -> str:
    return config.manifest_path.read_text(encoding="utf-8")


def load_manifest(config: DataModuleConfig) -> list[ManifestEntry]:
    return _MANIFEST.validate_json(read_manifest(config))


def get_manifest_size_bytes(entries: list[ManifestEntry]) -> int:
    return sum(entry.size_bytes for entry in entries)


def validate_new_file_name(config: DataModuleConfig, name: str) -> None:
    if Path(name).name != name or name in ("", ".."):
        raise ValueError(f"'{name}' is not a plain file name")
    if (config.files_dir / name).exists():
        raise ValueError(f"{_get_manifest_path(config, config.files_dir / name)} already exists")


def create_temporary_download_directory(config: DataModuleConfig) -> tempfile.TemporaryDirectory:
    # Inside the collected data folder, so that moving a download to its place does not copy it.
    return tempfile.TemporaryDirectory(dir=config.collected_data_dir)


def add_to_collected_data_directory(
    config: DataModuleConfig, path: Path, name: str, source_url: str, archive: str | None,
) -> tuple[str, list[ManifestEntry]]:
    target = config.files_dir / name
    target.parent.mkdir(exist_ok=True)
    path.rename(target)
    files = [target] if target.is_file() else sorted(file for file in target.rglob("*") if file.is_file())
    new_manifests = [
        ManifestEntry(
            path=_get_manifest_path(config, file),
            source_url=source_url,
            size_bytes=file.stat().st_size,
            sha256=get_file_hash(file),
            archive=archive,
        )
        for file in files
    ]
    _save_manifest(config, load_manifest(config) + new_manifests)
    return _get_manifest_path(config, target), new_manifests


def discard_from_collected_data_directory(config: DataModuleConfig, path: str) -> list[ManifestEntry]:
    path = path.rstrip("/")
    entries = load_manifest(config)
    discarded = [entry for entry in entries if entry.path == path or entry.path.startswith(f"{path}/")]
    if not discarded:
        raise ValueError(f"{path} is not in the manifest")
    remove_path(config.collected_data_dir / path)
    remove_empty_directories((config.collected_data_dir / path).parent, config.files_dir)
    _save_manifest(config, [entry for entry in entries if entry not in discarded])
    return discarded


def append_data_collection_review(config: DataModuleConfig, review: BaseModel) -> None:
    with config.reviews_path.open("a", encoding="utf-8") as file:
        file.write(review.model_dump_json() + "\n")


def _get_manifest_path(config: DataModuleConfig, path: Path) -> str:
    return path.relative_to(config.collected_data_dir).as_posix()


def _save_manifest(config: DataModuleConfig, entries: list[ManifestEntry]) -> None:
    with write_atomically(config.manifest_path) as temporary_path:
        temporary_path.write_bytes(_MANIFEST.dump_json(entries, indent=2, exclude_none=True))
