from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(kw_only=True)
class DataModuleConfig:
    FILES_DIRNAME: ClassVar[str] = "files"
    MANIFEST_FILENAME: ClassVar[str] = "manifest.json"
    REVIEWS_FILENAME: ClassVar[str] = "reviews.jsonl"
    OUTPUT_FILENAME: ClassVar[str] = "output.json"
    BYTES_PER_GB: ClassVar[int] = 1_000_000_000

    DEFAULT_MAX_STEPS: ClassVar[int] = 100
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.7
    DEFAULT_MAX_VALIDATION_RETRIES: ClassVar[int] = 3
    DEFAULT_MAX_REVIEWER_SEARCHES: ClassVar[int] = 3
    DEFAULT_USE_PROXY: ClassVar[bool] = True
    DEFAULT_LLM_RESPONSE_TIMEOUT: ClassVar[int] = 60 * 10
    DEFAULT_DOWNLOAD_FILE_TOOL_TIMEOUT: ClassVar[int] = 60
    DEFAULT_MAX_TOOL_RETRIES: ClassVar[int] = 5
    DEFAULT_STRATEGY: ClassVar[str] = "Download the most relevant, highest-quality sources first, then the rest of the relevant data."

    collected_data_dir: Path
    collector_model_name: str
    reviewer_model_name: str

    max_download_gb: float | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float = DEFAULT_TEMPERATURE
    max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES
    max_reviewer_searches: int = DEFAULT_MAX_REVIEWER_SEARCHES
    use_proxy: bool = DEFAULT_USE_PROXY
    llm_response_timeout: int = DEFAULT_LLM_RESPONSE_TIMEOUT
    download_file_tool_timeout: int = DEFAULT_DOWNLOAD_FILE_TOOL_TIMEOUT
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES

    @property
    def files_dir(self) -> Path:
        return self.collected_data_dir / self.FILES_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.collected_data_dir / self.MANIFEST_FILENAME

    @property
    def reviews_path(self) -> Path:
        return self.collected_data_dir / self.REVIEWS_FILENAME

    @property
    def output_path(self) -> Path:
        return self.collected_data_dir / self.OUTPUT_FILENAME

    @property
    def max_download_bytes(self) -> int | None:
        if self.max_download_gb is None:
            return None
        return int(self.max_download_gb * self.BYTES_PER_GB)
