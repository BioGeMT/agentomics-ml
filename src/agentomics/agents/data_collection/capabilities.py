from __future__ import annotations

from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, WebFetch, WebSearch
from pydantic_ai_harness import FileSystem

from agentomics.tools.web_fetch_tool import create_web_fetch_tool
from agentomics.utils.data_module_config import DataModuleConfig


def create_data_collection_capabilities(config: DataModuleConfig) -> list[AbstractCapability]:
    return [
        WebSearch(local=False),
        WebFetch(local=create_web_fetch_tool(config)),
        LocalWorkspace(config.collected_data_dir, read_only=True),
        FileSystem(read_only=True, tools=["read_file", "list_directory"]),
    ]
