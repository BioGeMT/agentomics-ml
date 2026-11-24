from pathlib import Path

from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from agentomics.runtime import paper_retrieval
from agentomics.runtime.paper_retrieval import prepare_papers


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


def test_prepare_papers_writes_page_text_and_image_paths(default_agent_config, tmp_path):
    paper_path = default_agent_config.fetched_papers_dir / "example.pdf"
    _write_paper(paper_path)
    _write_blank_paper(default_agent_config.fetched_papers_dir / "blank.pdf")
    outside = tmp_path / "outside.pdf"
    _write_blank_paper(outside)
    (default_agent_config.fetched_papers_dir / "escape.pdf").symlink_to(outside)

    root = prepare_papers(default_agent_config)

    documents = list(root.glob("*/document.md"))
    assert len(documents) == 2
    document_path = next(path for path in documents if path.parent.name.startswith("example-"))
    text = document_path.read_text()
    assert "## Page 2" in text
    assert "convolutional network" in text
    image_path = document_path.parent / "pages" / "page_0002.png"
    assert image_path.is_file()
    assert str(image_path.resolve()) in text
    blank_path = next(path for path in documents if path.parent.name.startswith("blank-"))
    assert "## Page 1" in blank_path.read_text()
    assert (blank_path.parent / "pages" / "page_0001.png").is_file()


def test_prepare_papers_omits_page_images_over_the_size_limit(default_agent_config, monkeypatch):
    _write_paper(default_agent_config.fetched_papers_dir / "example.pdf")
    monkeypatch.setattr(paper_retrieval, "MAX_PAGE_IMAGE_BYTES", 1)

    root = prepare_papers(default_agent_config)

    assert not list(root.rglob("*.png"))
    text = next(root.glob("*/document.md")).read_text()
    assert "Page image unavailable" in text
    assert "[Page image](" not in text
