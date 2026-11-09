from pathlib import Path

from pydantic import BaseModel, TypeAdapter

MANIFEST_FILE_NAME = "manifest.json"


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
    manifest_path = collected_data_directory / MANIFEST_FILE_NAME
    temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_path.write_bytes(_MANIFEST.dump_json(entries, indent=2, exclude_none=True))
    temporary_path.replace(manifest_path)
