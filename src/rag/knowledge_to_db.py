"""
End-to-end RAG knowledge pipeline.

Steps:
  1. Convert PDFs to markdown (Docling)
  2. Clean markdown (remove boilerplate sections and non-text placeholders)
  3. [TODO] Chunk cleaned markdown
  4. [TODO] Embed chunks and store in vector DB

Usage:
    python knowledge_to_db.py --dataset src/rag/raw_knowledge
"""

import argparse
import os
from pathlib import Path

# Prevent proxy from intercepting local Ollama/pgvector requests
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from pdfs_to_markdown import convert_all_pdfs
from clean_markdown import clean_all

RAG_DIR = Path(__file__).parent


def needs_update(source_dir: Path, output_dir: Path, suffix: str = ".md") -> bool:
    """Return True if any output file is missing or older than its source."""
    sources = list(source_dir.iterdir())
    if not sources:
        return False
    for src in sources:
        out = output_dir / f"{src.stem}{suffix}"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            return True
    return False


def step_convert_pdfs(pdf_dir: Path, processed_dir: Path) -> None:
    print("=" * 50)
    print("Step 1: PDF → Markdown (Docling)")
    print("=" * 50)
    if needs_update(pdf_dir, processed_dir):
        convert_all_pdfs(dataset_knowledge_dir=pdf_dir, output_dir=processed_dir)
    else:
        print("  All markdown files up to date, skipping.\n")


def step_clean_markdown(processed_dir: Path, cleaned_dir: Path) -> None:
    print("=" * 50)
    print("Step 2: Clean Markdown")
    print("=" * 50)
    if needs_update(processed_dir, cleaned_dir):
        clean_all(input_dir=processed_dir, output_dir=cleaned_dir)
    else:
        print("  All cleaned files up to date, skipping.\n")


def step_chunk(cleaned_dir: Path) -> None:
    print("=" * 50)
    print("Step 3: Chunk [TODO]")
    print("=" * 50)
    # TODO: split cleaned markdown into overlapping chunks
    # suitable for embedding (e.g. by section or fixed token window)
    print("  Not implemented yet.\n")


def step_embed_and_store(cleaned_dir: Path, db_dir: Path) -> None:
    print("=" * 50)
    print("Step 4: Embed + Store in Vector DB [TODO]")
    print("=" * 50)
    # TODO: embed chunks and persist to vector DB (e.g. ChromaDB)
    print("  Not implemented yet.\n")


def main():
    parser = argparse.ArgumentParser(description="Build RAG knowledge DB from a directory of PDFs.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RAG_DIR / "raw_knowledge",
        help="Path to directory containing source PDFs (default: src/rag/raw_knowledge)",
    )
    args = parser.parse_args()

    pdf_dir = args.dataset.resolve()
    dataset_name = pdf_dir.name

    processed_dir = RAG_DIR / "processed_knowledge" / dataset_name
    cleaned_dir   = RAG_DIR / "cleaned_knowledge"   / dataset_name
    db_dir        = RAG_DIR / "db"                  / dataset_name

    print(f"Dataset : {pdf_dir}")
    print(f"Processed: {processed_dir}")
    print(f"Cleaned  : {cleaned_dir}")
    print(f"DB       : {db_dir}\n")

    step_convert_pdfs(pdf_dir, processed_dir)
    step_clean_markdown(processed_dir, cleaned_dir)
    step_chunk(cleaned_dir)
    step_embed_and_store(cleaned_dir, db_dir)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
