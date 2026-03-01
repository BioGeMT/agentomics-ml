import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pdfs_to_markdown import convert_all_pdfs
from clean_markdown import clean_all

def needs_update(source_dir: Path, output_dir: Path, source_suffix: str, output_suffix: str = ".md") -> bool:
    """Return True if any output file is missing or older than its source."""
    sources = list(source_dir.glob(f"*{source_suffix}"))
    if not sources:
        return False
    for src in sources:
        out = output_dir / f"{src.stem}{output_suffix}"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            return True
    return False


def step_convert_pdfs(knowledge_dir: Path, processed_knowledge_dir: Path) -> None:
    print("=" * 50)
    print("Step 1: Knowledge → Markdown (using Docling)")
    print("=" * 50)
    if needs_update(source_dir=knowledge_dir, output_dir=processed_knowledge_dir, source_suffix=".pdf"):
        convert_all_pdfs(dataset_knowledge_dir=knowledge_dir, output_dir=processed_knowledge_dir)
    else:
        print("  All markdown files up to date, skipping.\n")


def step_clean_markdown(processed_dir: Path, cleaned_dir: Path) -> None:
    print("=" * 50)
    print("Step 2: Clean Markdown")
    print("=" * 50)
    if needs_update(processed_dir, cleaned_dir, source_suffix=".md"):
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


def step_embed_and_store(cleaned_dir: Path) -> None:
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
        help="Path to dataset directory"
    )
    args = parser.parse_args()

    dataset_dir = Path("datasets" / args.dataset).resolve()
    knowledge_dir = dataset_dir / "knowledge"
    processed_dir = knowledge_dir / "processed_knowledge"
    cleaned_dir = knowledge_dir / "cleaned_knowledge"

    step_convert_pdfs(knowledge_dir, processed_dir)
    step_clean_markdown(processed_dir, cleaned_dir)
    step_chunk(cleaned_dir)
    step_embed_and_store(cleaned_dir)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
