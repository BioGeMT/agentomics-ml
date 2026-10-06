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


def test_retrieves_and_inspects_relevant_paper_page(default_agent_config):
    paper_path = default_agent_config.fetched_papers_dir / "example.pdf"
    _write_paper(paper_path)
    _write_blank_paper(default_agent_config.fetched_papers_dir / "blank.pdf")

    retrieval_tool = create_tool(default_agent_config, "retrieve_paper_chunks")
    results = retrieval_tool.function(
        query="Which model is used for the zebrafish classifier?",
        max_results=1,
    )

    assert results[0]["paper"] == paper_path.name
    assert results[0]["page"] == 2
    assert "convolutional network" in results[0]["text"]
    assert Path(results[0]["markdown_path"]).is_file()
    assert Path(results[0]["markdown_path"]).with_name("chunks.json").is_file()
    assert Path(results[0]["image_path"]).is_file()

    inspect_tool = create_tool(default_agent_config, "inspect_image")
    inspected = inspect_tool.function(
        image_path=results[0]["image_path"],
        query="What model does this page describe?",
    )

    assert isinstance(inspected, ToolReturn)
    assert isinstance(inspected.content[1], BinaryContent)
    assert inspected.content[1].media_type == "image/png"
    assert retrieval_tool.function(
        query="zygomorphic quasar upholstery",
        max_results=1,
    ) == []


def test_inspect_image_rejects_files_outside_the_run(default_agent_config, tmp_path):
    image_path = tmp_path / "outside.png"
    image_path.write_bytes(b"not an accessible image")
    inspect_tool = create_tool(default_agent_config, "inspect_image")

    with pytest.raises(ModelRetry, match="must be inside"):
        inspect_tool.function(image_path=str(image_path), query="What is shown?")
