"""Statistics about the RAG knowledge stored in pgvector."""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src" / "rag") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "rag"))

from rag_utils.db_helpers import get_conn, TABLE  # noqa: E402

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(texts: list[str]) -> list[int]:
    return [len(ids) for ids in TOKENIZER.encode_batch(texts)]


def knowledge_stats() -> dict:
    """Print and return statistics about the knowledge table."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT source, section, content FROM {TABLE}")
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"Table {TABLE} is empty.")
        return {"total_chunks": 0}

    contents = [r[2] for r in rows]
    token_counts = _count_tokens(contents)
    char_counts = [len(c) for c in contents]

    per_source = defaultdict(lambda: {"chunks": 0, "tokens": 0, "chars": 0})
    per_section = defaultdict(int)
    for (source, section, _), tok, ch in zip(rows, token_counts, char_counts):
        per_source[source]["chunks"] += 1
        per_source[source]["tokens"] += tok
        per_source[source]["chars"] += ch
        per_section[section] += 1

    total_chunks = len(rows)
    total_tokens = sum(token_counts)
    total_chars = sum(char_counts)

    print("=" * 70)
    print(f"KNOWLEDGE DB STATS (table: {TABLE})")
    print("=" * 70)
    print(f"Total chunks:             {total_chunks}")
    print(f"Total tokens:             {total_tokens}")
    print(f"Total characters:         {total_chars}")
    print(f"Unique sources:           {len(per_source)}")
    print(f"Unique sections:          {len(per_section)}")
    print()
    print(f"Tokens per chunk   — mean: {statistics.mean(token_counts):.1f}, "
          f"median: {statistics.median(token_counts):.0f}, "
          f"min: {min(token_counts)}, max: {max(token_counts)}, "
          f"stdev: {statistics.pstdev(token_counts):.1f}")
    print(f"Chars  per chunk   — mean: {statistics.mean(char_counts):.1f}, "
          f"median: {statistics.median(char_counts):.0f}, "
          f"min: {min(char_counts)}, max: {max(char_counts)}")

    print()
    print("-" * 70)
    print(f"{'SOURCE':40} {'CHUNKS':>8} {'TOKENS':>10} {'TOK/CHUNK':>10}")
    print("-" * 70)
    for source in sorted(per_source):
        s = per_source[source]
        avg = s["tokens"] / s["chunks"] if s["chunks"] else 0
        print(f"{source[:40]:40} {s['chunks']:>8} {s['tokens']:>10} {avg:>10.1f}")

    return {
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "unique_sources": len(per_source),
        "unique_sections": len(per_section),
        "tokens_per_chunk": {
            "mean": statistics.mean(token_counts),
            "median": statistics.median(token_counts),
            "min": min(token_counts),
            "max": max(token_counts),
            "stdev": statistics.pstdev(token_counts),
        },
        "chars_per_chunk": {
            "mean": statistics.mean(char_counts),
            "median": statistics.median(char_counts),
            "min": min(char_counts),
            "max": max(char_counts),
        },
        "per_source": dict(per_source),
        "per_section": dict(per_section),
    }


if __name__ == "__main__":
    knowledge_stats()
