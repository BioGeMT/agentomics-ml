from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from .download_file_tool import DownloadDeps
from agentomics.runtime.collected_data_directory import discard_from_collected_data_directory, get_manifest_size_bytes
from agentomics.utils.data_module_config import DataModuleConfig


class DiscardFileResult(BaseModel):
    file_count: int
    used_bytes: int
    max_bytes: int | None


def create_discard_file_tool(config: DataModuleConfig) -> Tool[DownloadDeps]:
    return Tool(
        function=_discard_file,
        takes_ctx=True,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        name="discard_file",
        sequential=True,
    )


def _discard_file(ctx: RunContext[DownloadDeps], path: str) -> DiscardFileResult | str:
    """Delete a downloaded file that does not fit the spec, and remove it from the manifest.

    Args:
        path: The path of the file as the manifest records it, or of the folder an archive was unpacked into.
    """
    byte_budget = ctx.deps.byte_budget
    try:
        discarded = discard_from_collected_data_directory(ctx.deps.config, path)
    except ValueError as error:
        return f"Error: {error}"
    byte_budget.release(get_manifest_size_bytes(discarded))
    return DiscardFileResult(file_count=len(discarded), used_bytes=byte_budget.used_bytes, max_bytes=byte_budget.max_bytes)
