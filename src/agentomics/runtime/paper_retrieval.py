from __future__ import annotations

import hashlib
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from agentomics.utils.config import Config


def prepare_papers(config: Config) -> Path:
    config.processed_papers_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(config.fetched_papers_dir.glob("*.pdf")):
        with pdf_path.open("rb") as pdf_file:
            digest = hashlib.file_digest(pdf_file, "sha256").hexdigest()[:12]
        paper_dir = config.processed_papers_dir / f"{pdf_path.stem}-{digest}-v2"
        document_path = paper_dir / "document.md"
        if document_path.exists():
            continue

        pages = []
        with pdfium.PdfDocument(pdf_path) as document:
            for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
                text = (
                    (page.extract_text(extraction_mode="layout") or "").strip()
                    if page.get_contents() is not None
                    else ""
                )
                image_path = paper_dir / "pages" / f"page_{page_number:04d}.png"
                _render_page(document, page_number, image_path)
                pages.append(f"## Page {page_number}\n\n[Page image]({image_path.resolve()})\n\n{text}")

        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(
            f"# {pdf_path.name}\n\nSource PDF: {pdf_path.resolve()}\n\n"
            + "\n\n".join(pages)
            + "\n",
            encoding="utf-8",
        )

    return config.processed_papers_dir


def _render_page(document: pdfium.PdfDocument, page_number: int, image_path: Path) -> None:
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
