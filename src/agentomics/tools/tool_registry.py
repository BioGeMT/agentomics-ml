from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import weave
from pydantic_ai import Tool

from .bash_tool import create_bash_tool
from .inspect_image_tool import create_inspect_image_tool
from .replace_tool import create_replace_tool
from .retrieve_paper_chunks_tool import create_retrieve_paper_chunks_tool
from .run_python_tool import create_run_python_tool
from .write_python_tool import create_write_python_tool

from agentomics.utils.config import Config

TOOL_FACTORY_REGISTRY: dict[str, Callable] = {
    "bash": create_bash_tool,
    "inspect_image": create_inspect_image_tool,
    "retrieve_paper_chunks": create_retrieve_paper_chunks_tool,
    "write_python": create_write_python_tool,
    "run_python": create_run_python_tool,
    "replace": create_replace_tool,
}

def create_tool(config: Config, tool_id: str) -> Tool[Any]:
    tool = TOOL_FACTORY_REGISTRY[tool_id](config)
    tool.function = weave.op(tool.function, call_display_name=tool.name)
    return tool

def create_tools(config: Config, tool_ids: Iterable[str]) -> list[Tool[Any]]:
    return [create_tool(config, tool_id) for tool_id in tool_ids]

def get_tool_names(config: Config, tool_ids: Iterable[str]) -> list[str]:
    return [tool.name for tool in create_tools(config, tool_ids)]
