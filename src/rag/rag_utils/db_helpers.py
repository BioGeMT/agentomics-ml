import yaml
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG_PATH = Path(__file__).parent / "db_config.yaml"
TABLE = "knowledge"
EMBED_DIMS = 4096

def read_db_config(config_path: Path = DB_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
    
def get_conn(config_path: Path = DB_CONFIG_PATH):
    cfg = read_db_config(config_path)
    return psycopg2.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
    )

def setup_schema(conn) -> None:
    """Create the vector extension and knowledge table if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id        SERIAL PRIMARY KEY,
                source    TEXT,
                section   TEXT,
                content   TEXT NOT NULL,
                embedding vector({EMBED_DIMS})
            )
        """)
    conn.commit()


def build_index(conn) -> None:
    """Build index — skipped for >2000 dims (pgvector limitation for hnsw/ivfflat)."""
    pass

def store_chunks(conn, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Insert chunks with their embeddings into the knowledge table."""
    rows = [
        (c["source"], c["section"], c["content"], str(v))
        for c, v in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {TABLE} (source, section, content, embedding) VALUES %s",
            rows,
        )
    conn.commit()

def retrieve(conn, query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """Return the top_k most similar chunks to the query embedding.

    Each result includes the stored embedding as a list[float] so callers can
    run downstream re-ranking (e.g. MMR) without re-embedding the chunks.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT source, section, content, embedding,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (str(query_embedding), str(query_embedding), top_k))
        rows = cur.fetchall()

    def _parse_vector(v) -> list[float]:
        # pgvector returns the vector as a string like "[0.1,0.2,...]"
        if isinstance(v, str):
            return [float(x) for x in v.strip("[]").split(",")]
        return list(v)

    return [
        {
            "source": r[0],
            "section": r[1],
            "content": r[2],
            "embedding": _parse_vector(r[3]),
            "score": r[4],
        }
        for r in rows
    ]

def is_up_to_date(conn, chunks: list[dict]) -> bool:
    """Return True if the DB already contains exactly the same chunks (by count per source)."""
    expected = {}
    for c in chunks:
        expected[c["source"]] = expected.get(c["source"], 0) + 1
    with conn.cursor() as cur:
        cur.execute(f"SELECT source, COUNT(*) FROM {TABLE} GROUP BY source")
        actual = {row[0]: row[1] for row in cur.fetchall()}

    if not expected == actual:
        print("Knowledge not up to date, rebuilding DB.")
    return expected == actual


def clear_table(conn) -> None:
    """Delete all rows from the knowledge table."""
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {TABLE}")
    conn.commit()
