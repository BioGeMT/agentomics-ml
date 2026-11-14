import asyncio
from types import SimpleNamespace

from pydantic_ai import BinaryContent

from agentomics.tools import web_fetch_tool
from agentomics.tools.web_fetch_tool import create_web_fetch_tool


def test_page_reading_says_when_a_url_is_a_file_the_model_cannot_take(data_module_config, monkeypatch):
    async def fetch_archive(url: str) -> BinaryContent:
        return BinaryContent(data=b"PK", media_type="application/zip")

    monkeypatch.setattr(web_fetch_tool, "web_fetch_tool", lambda: SimpleNamespace(function=fetch_archive))

    result = asyncio.run(create_web_fetch_tool(data_module_config).function("https://example.org/data.zip"))

    assert "application/zip" in result and "not a page" in result
