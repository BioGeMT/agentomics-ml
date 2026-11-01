from pathlib import Path

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import BinaryContent, ToolReturn
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from agentomics.tools.tool_registry import create_tool


def _write_paper(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(path))
    document.drawString(72, 720, "Background information about protein binding.")
    document.showPage()
    document.drawString(72, 720, "The zebrafish classifier uses a convolutional network.")
    document.save()


def _write_blank_paper(path: Path) -> None:
    document = PdfWriter()
    document.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        document.write(output)


def test_searches_reads_and_inspects_papers(default_agent_config):
    paper_path = default_agent_config.fetched_papers_dir / "example.pdf"
    _write_paper(paper_path)
    _write_blank_paper(default_agent_config.fetched_papers_dir / "blank.pdf")

    entries = create_tool(default_agent_config, "ls").function()
    assert len(entries) == 2

    documents = create_tool(default_agent_config, "glob").function(pattern="**/*.md")
    assert len(documents) == 2

    matches = create_tool(default_agent_config, "grep").function(
        pattern="convolutional network",
    )
    assert len(matches) == 1
    text = create_tool(default_agent_config, "read_file").function(
        path=matches[0]["path"],
        offset=matches[0]["line"],
        limit=1,
    )
    assert "convolutional network" in text

    image_path = create_tool(default_agent_config, "render_paper_page").function(
        pdf_path=str(paper_path),
        page_number=2,
    )
    assert Path(image_path).is_file()

    inspect_tool = create_tool(default_agent_config, "inspect_image")
    inspected = inspect_tool.function(
        image_path=image_path,
        query="What model does this page describe?",
    )

    assert isinstance(inspected, ToolReturn)
    assert isinstance(inspected.content[1], BinaryContent)
    assert inspected.content[1].media_type == "image/png"


def test_paper_tools_reject_files_outside_their_directories(default_agent_config, tmp_path):
    image_path = tmp_path / "outside.png"
    image_path.write_bytes(b"not an accessible image")

    with pytest.raises(ModelRetry, match="processed papers"):
        create_tool(default_agent_config, "read_file").function(path=str(image_path))
    with pytest.raises(ModelRetry, match="must be inside"):
        create_tool(default_agent_config, "inspect_image").function(
            image_path=str(image_path),
            query="What is shown?",
        )
