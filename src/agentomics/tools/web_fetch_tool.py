from __future__ import annotations

from pydantic_ai import BinaryContent, Tool
from pydantic_ai.common_tools.web_fetch import WebFetchResult, web_fetch_tool

from agentomics.utils.data_module_config import DataModuleConfig


# TODO: drop this tool and use pydantic-ai's own web fetch once
# https://github.com/pydantic/pydantic-ai/issues/11829 is fixed.
def create_web_fetch_tool(config: DataModuleConfig) -> Tool:
    return Tool(
        function=_web_fetch,
        takes_ctx=False,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        name="web_fetch",
    )


async def _web_fetch(url: str) -> WebFetchResult | BinaryContent | str:
    """Fetches the content of a web page at the given URL and returns it as markdown or binary content.

    Args:
        url: The URL to fetch.
    """
    result = await web_fetch_tool().function(url)
    # pydantic-ai passes a fetched non-text file to the model as is, and its OpenAI models raise RuntimeError on
    # most types other than images and PDFs, which ends the run.
    if isinstance(result, BinaryContent) and not (result.is_image or result.media_type == "application/pdf"):
        return f"{url} is a {result.media_type} file of {len(result.data)} bytes, not a page. Download it to look inside."
    return result
