from __future__ import annotations

import hashlib
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from agentomics.utils.config import Config

# Leave room for base64 encoding under a 5 MB image limit.
MAX_PAGE_IMAGE_BYTES = 3 * 1024 * 1024


def prepare_papers(config: Config) -> Path:
    config.processed_papers_dir.mkdir(parents=True, exist_ok=True)
    fetched_dir = config.fetched_papers_dir.resolve()
    for pdf_path in sorted(config.fetched_papers_dir.glob("*.pdf")):
        if not pdf_path.resolve().is_relative_to(fetched_dir):
            continue
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
                image_line = (
                    f"[Page image]({image_path.resolve()})"
                    if _render_page(document, page_number, image_path)
                    else "Page image unavailable: exceeds the image size limit."
                )
                pages.append(f"## Page {page_number}\n\n{image_line}\n\n{text}")

        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(
            f"# {pdf_path.name}\n\nSource PDF: {pdf_path.resolve()}\n\n"
            + "\n\n".join(pages)
            + "\n",
            encoding="utf-8",
        )

    return config.processed_papers_dir


def _render_page(document: pdfium.PdfDocument, page_number: int, image_path: Path) -> bool:
    page = document[page_number - 1]
    try:
        for scale in (2, 1):
            bitmap = page.render(scale=scale)
            try:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                bitmap.to_pil().save(image_path, format="PNG")
            finally:
                bitmap.close()
            if image_path.stat().st_size <= MAX_PAGE_IMAGE_BYTES:
                return True
        image_path.unlink()
        return False
    finally:
        page.close()
