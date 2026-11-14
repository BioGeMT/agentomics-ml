from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import httpx2
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from agentomics.runtime.archives import get_archive_suffix, unpack_archive
from agentomics.runtime.collected_data_directory import (
    add_to_collected_data_directory,
    create_temporary_download_directory,
    get_manifest_size_bytes,
    validate_new_file_name,
)
from agentomics.runtime.filesystem import ByteBudget, write_chunks
from agentomics.utils.data_module_config import DataModuleConfig
from agentomics.utils.exceptions import DownloadError

# Other kinds of sources (cloud buckets, ftp, toolkits like SRA or Synapse) each need their own way to download.
SUPPORTED_URL_PREFIXES = ("http://", "https://")


@dataclass(kw_only=True)
class DownloadDeps:
    config: DataModuleConfig
    byte_budget: ByteBudget


@dataclass
class DownloadItem:
    url: str
    path: Path
    content_type: str | None


class DownloadFileResult(BaseModel):
    path: str
    file_count: int
    content_type: str | None
    used_bytes: int
    max_bytes: int | None


def create_download_file_tool(config: DataModuleConfig) -> Tool[DownloadDeps]:
    return Tool(
        function=_download_file,
        takes_ctx=True,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        name="download_file",
        sequential=True,
    )


def _download_file(ctx: RunContext[DownloadDeps], url: str, filename: str) -> DownloadFileResult | str:
    """Download a file from a public URL into the collected data folder and record it in the manifest.

    If the file is a compressed archive, it is unpacked under the file name without its archive suffix,
    and the archive is removed.

    Args:
        url: The http or https URL of the file.
        filename: The name to save it under: a plain file name, without folders.
    """
    config, byte_budget = ctx.deps.config, ctx.deps.byte_budget
    archive_suffix = get_archive_suffix(filename)
    name = filename.removesuffix(archive_suffix or "")
    with create_temporary_download_directory(config) as temporary_directory:
        try:
            validate_new_file_name(config, name)
            # The download and the unpacking count against copies of the budget: a download that fails leaves the
            # budget as it was, and an unpacked archive counts only with the bytes of its files.
            download_item = _download_url(
                url, Path(temporary_directory) / filename, config.download_file_tool_timeout, replace(byte_budget),
            )
            path = download_item.path
            if archive_suffix:
                path = download_item.path.with_name(name)
                unpack_archive(download_item.path, path, replace(byte_budget))
        except (DownloadError, ValueError) as error:
            return f"Error: {error}"
        stored_path, new_manifests = add_to_collected_data_directory(
            config, path, name, download_item.url, archive=filename if archive_suffix else None,
        )
    byte_budget.consume(get_manifest_size_bytes(new_manifests))
    return DownloadFileResult(
        path=stored_path,
        file_count=len(new_manifests),
        content_type=download_item.content_type,
        used_bytes=byte_budget.used_bytes,
        max_bytes=byte_budget.max_bytes,
    )


def _download_url(url: str, path: Path, timeout: int, byte_budget: ByteBudget) -> DownloadItem:
    if not url.startswith(SUPPORTED_URL_PREFIXES):
        raise DownloadError("only http and https URLs can be downloaded. Try the same address over https if the host offers it")
    try:
        with httpx2.Client(follow_redirects=True, timeout=timeout) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            if "content-length" in response.headers:
                # Refuse a file that cannot fit before downloading it.
                replace(byte_budget).consume(int(response.headers["content-length"]))
            write_chunks(response.iter_bytes(), path, byte_budget)
    # UnicodeError: the host name cannot be encoded, for example because a label is empty.
    except (httpx2.HTTPError, httpx2.InvalidURL, UnicodeError) as error:
        raise DownloadError(f"the download failed: {error}") from error
    if path.stat().st_size == 0:
        raise DownloadError("the download was empty")
    return DownloadItem(url, path, response.headers.get("content-type"))
