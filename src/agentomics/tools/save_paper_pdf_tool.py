from typing import Any
from pathlib import Path

from pydantic_ai import Tool
from pydantic_ai.messages import BinaryContent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

from agentomics.utils.config import Config


def create_save_paper_pdf_tool(config: Config) -> Tool[Any]:
    fetch = web_fetch_tool(timeout=config.web_fetch_timeout).function

    async def save_paper_pdf(url: str, filename: str) -> str:
        """Downloads the PDF at the given URL and saves it in the papers folder.

        Args:
            url: Direct URL to the PDF.
            filename: Name to save it under, ending in .pdf. Directories are ignored;
                the tool decides where the file goes.
        """
        response = await fetch(url)
        if not isinstance(response, BinaryContent) or not is_pdf(response.data):
            raise ModelRetry(
                f"{url} did not return a PDF. Some sites serve an HTML download page instead; "
                "find the direct PDF link and retry."
            )

        papers_dir = config.fetched_papers_dir
        papers_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = papers_dir / Path(filename).name
        pdf_path.write_bytes(response.data)
        return f"Saved {len(response.data)} bytes to {pdf_path}"

    return Tool(save_paper_pdf, name="save_paper_pdf")

def is_pdf(data: bytes) -> bool:
    """A PDF always starts with the bytes `%PDF-`, whatever content type the server claims."""
    return data.startswith(b"%PDF-")
