"""
Smoke test for the RAG stack: chunk a real document, embed with
Qwen3-Embedding-8B (via Ollama), store in pgvector, retrieve by similarity.

Usage:
    conda run -n agentomics-env python src/rag/test_rag_stack.py
"""

import os
import re

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import requests
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/embed"
EMBED_MODEL = "ryanshillington/Qwen3-Embedding-8B:latest"
EMBED_DIMS  = 4096

PG_HOST     = "localhost"
PG_PORT     = 54320
PG_DB       = "rag"
PG_USER     = "rag"
PG_PASSWORD = "rag"

TABLE = "rag_test"

SOURCE_DOC = Path(__file__).parent / "cleaned_knowledge" / "mirbench.md"

MIN_CHUNK_CHARS = 80

QUERY = "Which model performed best on the Hejret 2023 test set?"

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "input": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_markdown(text: str) -> list[dict]:
    """
    Section-aware paragraph chunking.

    Splits on ## headings, then on blank lines within each section.
    Each chunk is prefixed with its section heading so the embedding
    captures both local content and document context.
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
                "source":  SOURCE_DOC.stem,
                "section": current_section,
                "content": f"[{current_section}]\n{para}",
            })

    return chunks

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Chunk the document
    text = SOURCE_DOC.read_text(encoding="utf-8")
    chunks = chunk_markdown(text)
    print(f"Document : {SOURCE_DOC.name}")
    print(f"Chunks   : {len(chunks)}\n")

    # 2. Set up schema
    conn = get_conn()
    cur = conn.cursor()

    print("Setting up schema...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cur.execute(f"""
        CREATE TABLE {TABLE} (
            id        SERIAL PRIMARY KEY,
            source    TEXT,
            section   TEXT,
            content   TEXT NOT NULL,
            embedding vector({EMBED_DIMS})
        )
    """)
    conn.commit()

    # 3. Embed and insert (batch all at once — Ollama handles it)
    print(f"Embedding {len(chunks)} chunks...")
    contents = [c["content"] for c in chunks]
    vecs = embed(contents)
    print(f"  Done.\n")

    rows = [
        (c["source"], c["section"], c["content"], str(v))
        for c, v in zip(chunks, vecs)
    ]
    execute_values(
        cur,
        f"INSERT INTO {TABLE} (source, section, content, embedding) VALUES %s",
        rows,
    )
    conn.commit()

    # 4. Query by similarity
    print(f"Query: '{QUERY}'")
    query_vec = embed([QUERY])[0]
    cur.execute(f"""
        SELECT section, content, 1 - (embedding <=> %s::vector) AS score
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT 3
    """, (str(query_vec), str(query_vec)))

    print("\nTop 3 results:")
    for i, (section, content, score) in enumerate(cur.fetchall(), 1):
        preview = content.replace("\n", " ")[:120]
        print(f"  {i}. [{score:.3f}] ({section})")
        print(f"       {preview}...")

    cur.close()
    conn.close()
    print("\nStack test passed.")


if __name__ == "__main__":
    main()
