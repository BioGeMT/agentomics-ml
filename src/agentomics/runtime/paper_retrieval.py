from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agentomics.utils.config import Config


CHUNK_WORDS = 350
CHUNK_OVERLAP_WORDS = 50


def retrieve_paper_chunks(
    config: Config,
    query: str,
    max_results: int,
) -> list[dict[str, str | int | float]]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    chunks: list[dict[str, str | int]] = []
    for pdf_path in sorted(config.fetched_papers_dir.glob("*.pdf")):
        paper_dir, paper_chunks = _prepare_pdf(config, pdf_path)
        chunks.extend(
            {
                "paper": pdf_path.name,
                **chunk,
                "markdown_path": str(paper_dir / "document.md"),
            }
            for chunk in paper_chunks
        )

    if not chunks:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    chunk_matrix = vectorizer.fit_transform(str(chunk["text"]) for chunk in chunks)
    scores = cosine_similarity(vectorizer.transform([query]), chunk_matrix).ravel()
    selected = [index for index in scores.argsort()[::-1] if scores[index] > 0][:max_results]

    results: list[dict[str, str | int | float]] = []
    for index in selected:
        chunk = chunks[int(index)]
        pdf_path = config.fetched_papers_dir / str(chunk["paper"])
        page_number = int(chunk["page"])
        page_image_path = _page_image_path(
            Path(str(chunk["markdown_path"])).parent,
            page_number,
        )
        if not page_image_path.exists():
            render_pdf_page(pdf_path, page_number, page_image_path)
        results.append(
            {
                **chunk,
                "score": float(scores[index]),
                "image_path": str(page_image_path),
            }
        )
    return results


def render_pdf_page(
    pdf_path: Path,
    page_number: int,
    image_path: Path,
) -> None:
    document = pdfium.PdfDocument(pdf_path)
    if not 1 <= page_number <= len(document):
        document.close()
        raise ValueError(f"page {page_number} is outside {pdf_path.name}")

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


def _prepare_pdf(
    config: Config,
    pdf_path: Path,
) -> tuple[Path, list[dict[str, str | int]]]:
    with pdf_path.open("rb") as pdf_file:
        digest = hashlib.file_digest(pdf_file, "sha256").hexdigest()[:12]
    paper_dir = config.processed_papers_dir / f"{pdf_path.stem}-{digest}"
    paper_dir.mkdir(parents=True, exist_ok=True)
    document_path = paper_dir / "document.md"
    chunks_path = paper_dir / "chunks.json"
    if document_path.exists() and chunks_path.exists():
        return paper_dir, json.loads(chunks_path.read_text(encoding="utf-8"))

    document_pages: list[str] = []
    chunks: list[dict[str, str | int]] = []
    for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = (
            (page.extract_text(extraction_mode="layout") or "").strip()
            if page.get_contents() is not None
            else ""
        )
        document_pages.append(f"## Page {page_number}\n\n{text}")
        chunks.extend(
            {
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk,
            }
            for chunk_number, chunk in enumerate(_split_text(text), start=1)
        )

    document_path.write_text(
        f"# {pdf_path.name}\n\n" + "\n\n".join(document_pages) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    return paper_dir, chunks


def _split_text(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return [
        " ".join(words[start : start + CHUNK_WORDS])
        for start in range(0, len(words), step)
    ]


def _page_image_path(paper_dir: Path, page_number: int) -> Path:
    return paper_dir / "pages" / f"page_{page_number:04d}.png"
