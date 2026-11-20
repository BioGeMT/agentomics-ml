import asyncio
from pathlib import Path

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.test import TestModel
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from agentomics.runtime.run_lifecycle import run_agentomics
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

    document_path = default_agent_config.processed_papers_dir / matches[0]["path"]
    image_path = document_path.parent / "pages" / "page_0002.png"
    assert Path(image_path).is_file()
    assert str(image_path.resolve()) in document_path.read_text()

    inspect_tool = create_tool(default_agent_config, "inspect_image")
    inspected = inspect_tool.function(
        image_path=str(image_path),
    )

    assert isinstance(inspected, BinaryContent)
    assert inspected.media_type == "image/png"
    assert inspected.data == image_path.read_bytes()


def test_paper_tools_reject_files_outside_their_directories(default_agent_config, tmp_path):
    image_path = tmp_path / "outside.png"
    image_path.write_bytes(b"not an accessible image")
    default_agent_config.processed_papers_dir.mkdir(parents=True)
    (default_agent_config.processed_papers_dir / "escape.md").symlink_to(image_path)
    assert create_tool(default_agent_config, "grep").function(pattern="accessible") == []

    with pytest.raises(ModelRetry, match="processed papers"):
        create_tool(default_agent_config, "read_file").function(path=str(image_path))
    with pytest.raises(ModelRetry, match="must be inside") as error:
        create_tool(default_agent_config, "inspect_image").function(
            image_path=str(image_path),
        )
    assert str(default_agent_config.processed_papers_dir.resolve()) in str(error.value)

    oversized_image = default_agent_config.processed_papers_dir / "oversized.png"
    with oversized_image.open("wb") as image:
        image.truncate(4 * 1024 * 1024)
    with pytest.raises(ModelRetry, match="exceeds"):
        create_tool(default_agent_config, "inspect_image").function(image_path=str(oversized_image))


def test_run_rejects_text_only_models_with_image_inspection(default_agent_config):
    model = TestModel(profile={"supports_image_input": False})
    with pytest.raises(ValueError, match="must support image input"):
        asyncio.run(run_agentomics(default_agent_config, model, model, model, None))
