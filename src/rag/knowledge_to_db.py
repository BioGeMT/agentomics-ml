import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pdfs_to_markdown import convert_all_pdfs
from clean_markdown import clean_all
from embed import embed
from rag_utils.db_helpers import get_conn, setup_schema, store_chunks, clear_table, build_index, is_up_to_date

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


MIN_CHUNK_CHARS = 150


def chunk_markdown(text: str, source: str) -> list[dict]:
    """
    Section-aware semantic chunking.

    Splits on ## headings, then on blank lines within each section.
    Each chunk is prefixed with its source and section so the embedding
    captures document context even for short paragraphs.
    """
    chunks = []
    current_section = "Preamble"

    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue

        if lines[0].startswith("##"):
            current_section = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = part.strip()

        paragraphs = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]

        for para in paragraphs:
            if len(para) < MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "source": source,
                "section": current_section,
                "content": f"[{source}] [{current_section}]\n{para}",
            })

    return chunks


def step_chunk(cleaned_dir: Path) -> list[dict]:
    print("=" * 50)
    print("Step 3: Semantic Chunking")
    print("=" * 50)
    all_chunks = []
    for md_path in sorted(cleaned_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=md_path.stem)
        all_chunks.extend(chunks)
        print(f"  {md_path.name}: {len(chunks)} chunks")
    print(f"  Total: {len(all_chunks)} chunks\n")
    return all_chunks

def step_embed_and_store(chunks: list[dict]) -> None:
    print("=" * 50)
    print("Step 4: Embed + Store in Vector DB")
    print("=" * 50)
    conn = get_conn()
    setup_schema(conn)

    if is_up_to_date(conn, chunks):
        print("  DB already up to date, skipping.\n")
        conn.close()
        return

    clear_table(conn)
    contents = [c["content"] for c in chunks]
    print(f"  Embedding {len(chunks)} chunks...")
    embeddings = embed(contents)
    store_chunks(conn, chunks, embeddings)
    build_index(conn)
    conn.close()
    print(f"  Stored {len(chunks)} chunks in pgvector.\n")


def main():
    parser = argparse.ArgumentParser(description="Build RAG knowledge DB from a directory of PDFs.")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset directory"
    )
    args = parser.parse_args()

    dataset_dir = (Path("datasets") / args.dataset).resolve()
    knowledge_dir = dataset_dir / "knowledge"

    if not knowledge_dir.exists():
        print(f"No knowledge directory found at {knowledge_dir}, skipping RAG preparation.")
        return

    processed_dir = knowledge_dir / "processed_knowledge"
    cleaned_dir = knowledge_dir / "cleaned_knowledge"

    step_convert_pdfs(knowledge_dir, processed_dir)
    step_clean_markdown(processed_dir, cleaned_dir)
    chunks = step_chunk(cleaned_dir)
    step_embed_and_store(chunks)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
