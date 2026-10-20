from __future__ import annotations

import hashlib
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from agentomics.utils.config import Config


def prepare_papers(config: Config) -> Path:
    for pdf_path in sorted(config.fetched_papers_dir.glob("*.pdf")):
        document_path = _paper_dir(config, pdf_path) / "document.md"
        if document_path.exists():
            continue

        pages = []
        for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
            text = (
                (page.extract_text(extraction_mode="layout") or "").strip()
                if page.get_contents() is not None
                else ""
            )
            pages.append(f"## Page {page_number}\n\n{text}")

        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(
            f"# {pdf_path.name}\n\nSource PDF: {pdf_path.resolve()}\n\n"
            + "\n\n".join(pages)
            + "\n",
            encoding="utf-8",
        )

    return config.processed_papers_dir


def render_pdf_page(config: Config, pdf_path: str, page_number: int) -> Path:
    source = Path(pdf_path).resolve()
    if not source.is_relative_to(config.fetched_papers_dir.resolve()) or not source.is_file():
        raise ValueError(f"Paper must be inside {config.fetched_papers_dir}: {pdf_path}")

    image_path = _paper_dir(config, source) / "pages" / f"page_{page_number:04d}.png"
    if image_path.exists():
        return image_path

    document = pdfium.PdfDocument(source)
    if not 1 <= page_number <= len(document):
        document.close()
        raise ValueError(f"page {page_number} is outside {source.name}")

    page = document[page_number - 1]
    try:
        bitmap = page.render(scale=2)
        try:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            bitmap.to_pil().save(image_path, format="PNG")
        finally:
            bitmap.close()
    finally:
        page.close()
        document.close()
    return image_path


def _paper_dir(config: Config, pdf_path: Path) -> Path:
    with pdf_path.open("rb") as pdf_file:
        digest = hashlib.file_digest(pdf_file, "sha256").hexdigest()[:12]
    return config.processed_papers_dir / f"{pdf_path.stem}-{digest}"
