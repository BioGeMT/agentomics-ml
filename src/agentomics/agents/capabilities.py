from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, PrepareTools
from pydantic_ai.messages import ModelResponse
from pydantic_ai_harness import FileSystem

from agentomics.utils.config import Config


def create_read_only_file_capabilities(
    config: Config, root_dir: Path, max_tool_calls: int, max_context_tokens: int,
) -> list[AbstractCapability]:
    async def hide_tools_after_reading_limits(ctx, tool_defs):
        last_response = next((message for message in reversed(ctx.messages) if isinstance(message, ModelResponse)), None)
        last_request_input_tokens = last_response.usage.input_tokens if last_response is not None else 0
        # With no tools offered, the next request must give the final output.
        if ctx.usage.tool_calls >= max_tool_calls or last_request_input_tokens >= max_context_tokens:
            return []
        return tool_defs

    return [
        LocalWorkspace(root_dir, read_only=True),
        FileSystem(
            read_only=True,
            tools=["list_directory", "search_files", "read_file"],
            max_read_chars=config.read_only_file_max_read_chars,
            max_search_results=config.read_only_file_max_search_results,
            max_list_results=config.read_only_file_max_list_results,
            content_hashes=False,
        ),
        PrepareTools(hide_tools_after_reading_limits),
    ]
