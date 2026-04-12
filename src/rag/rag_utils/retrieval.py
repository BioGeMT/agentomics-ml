"""Shared RAG retrieval helpers.

Used by:
- src/agents/architecture.py (upfront batch retrieval in `rag` mode)
- src/tools/knowledge_tool.py (on-demand single-query retrieval in `rag_od` mode)

Keeping this logic in one place prevents the two call sites from drifting.
"""
import time

from pydantic import BaseModel

try:
    from rag.embed import embed as rag_embed
    from rag.rag_utils.db_helpers import get_conn as rag_get_conn, retrieve as rag_retrieve
    RAG_AVAILABLE = True
    RAG_IMPORT_ERROR = None
except ImportError as _rag_import_error:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = _rag_import_error


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mmr_select(candidates: list, final_k: int, lambda_: float = 0.6) -> list:
    """Maximal Marginal Relevance selection.

    Iteratively picks the candidate that maximises
        lambda * relevance - (1 - lambda) * max_sim_to_already_selected
    producing a diverse, relevant subset.
    """
    if not candidates:
        return []
    remaining = list(candidates)
    selected = [max(remaining, key=lambda c: c["score"])]
    remaining.remove(selected[0])

    while remaining and len(selected) < final_k:
        def mmr_score(cand):
            max_sim = max(_cosine(cand["embedding"], s["embedding"]) for s in selected)
            return lambda_ * cand["score"] - (1 - lambda_) * max_sim

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected


class RagRetrievalLog(BaseModel):
    """Tier-1 telemetry for a single RAG retrieval call. Serialised to the
    iteration report via save_step_output and traced by weave."""
    status: str                        # "ok" | "no_queries" | "rag_unavailable" | "error"
    error: str | None = None
    n_queries: int
    per_query_k: int
    final_k: int
    mmr_lambda: float
    n_candidates_raw: int              # total chunks retrieved across all queries (with duplicates)
    n_candidates_unique: int           # after exact-content dedup
    n_selected: int                    # after MMR (should be min(final_k, n_candidates_unique))
    source_distribution: dict          # {source: count} in the selected set
    per_query_top1_scores: list        # [{"query": str, "top1_score": float|None}]
    selected_chunks: list              # [{"source","section","score","preview"}]
    total_latency_ms: float


def _empty_log(
    status: str,
    n_queries: int,
    per_query_k: int,
    final_k: int,
    mmr_lambda: float,
    started_at: float,
    error: str | None = None,
) -> RagRetrievalLog:
    return RagRetrievalLog(
        status=status,
        error=error,
        n_queries=n_queries,
        per_query_k=per_query_k,
        final_k=final_k,
        mmr_lambda=mmr_lambda,
        n_candidates_raw=0,
        n_candidates_unique=0,
        n_selected=0,
        source_distribution={},
        per_query_top1_scores=[],
        selected_chunks=[],
        total_latency_ms=round((time.time() - started_at) * 1000, 1),
    )


def run_rag_retrieval(
    queries: list,
    per_query_k: int = 6,
    final_k: int = 7,
    mmr_lambda: float = 0.6,
) -> tuple[str, RagRetrievalLog]:
    """Embed queries, retrieve a wide candidate pool, MMR-rerank for diversity,
    return the formatted context string plus a telemetry log.

    Accepts either a list of queries (batch mode) or a single-element list
    (on-demand mode from the knowledge tool).
    """
    start = time.time()
    n_q = len(queries) if queries else 0

    if not RAG_AVAILABLE:
        print(
            f"[RAG] WARNING: retrieval requested but RAG imports failed. "
            f"ImportError: {RAG_IMPORT_ERROR}"
        )
        return "", _empty_log("rag_unavailable", n_q, per_query_k, final_k, mmr_lambda, start, str(RAG_IMPORT_ERROR))
    if not queries:
        return "", _empty_log("no_queries", n_q, per_query_k, final_k, mmr_lambda, start)

    try:
        conn = rag_get_conn()
        per_query_candidates = []  # list[(query, list[chunk])] — preserves which query found what
        for query in queries:
            vec = rag_embed([query])[0]
            results = rag_retrieve(conn, vec, top_k=per_query_k)
            per_query_candidates.append((query, results))
        conn.close()

        all_chunks = [c for _, chunks in per_query_candidates for c in chunks]

        # Deduplicate by exact content before MMR so pairwise sim isn't 1.0
        seen = set()
        unique_candidates = []
        for r in all_chunks:
            if r["content"] not in seen:
                seen.add(r["content"])
                unique_candidates.append(r)

        if not unique_candidates:
            log = _empty_log("ok", n_q, per_query_k, final_k, mmr_lambda, start)
            log.n_candidates_raw = len(all_chunks)
            log.per_query_top1_scores = [
                {"query": q, "top1_score": (chunks[0]["score"] if chunks else None)}
                for q, chunks in per_query_candidates
            ]
            return "", log

        selected = _mmr_select(unique_candidates, final_k=final_k, lambda_=mmr_lambda)

        lines = ["\n## Retrieved Knowledge\n"]
        lines.append("The following excerpts from scientific literature are relevant to your task:\n")
        for i, r in enumerate(selected, 1):
            lines.append(f"### [{i}] {r['source']} — {r['section']} (relevance: {r['score']:.2f})")
            lines.append(r["content"].split("\n", 1)[-1].strip())  # skip [source][section] prefix line
            lines.append("")
        context = "\n".join(lines)

        source_distribution: dict = {}
        for r in selected:
            source_distribution[r["source"]] = source_distribution.get(r["source"], 0) + 1

        log = RagRetrievalLog(
            status="ok",
            n_queries=n_q,
            per_query_k=per_query_k,
            final_k=final_k,
            mmr_lambda=mmr_lambda,
            n_candidates_raw=len(all_chunks),
            n_candidates_unique=len(unique_candidates),
            n_selected=len(selected),
            source_distribution=source_distribution,
            per_query_top1_scores=[
                {"query": q, "top1_score": (chunks[0]["score"] if chunks else None)}
                for q, chunks in per_query_candidates
            ],
            selected_chunks=[
                {
                    "source": r["source"],
                    "section": r["section"],
                    "score": round(r["score"], 3),
                    "preview": r["content"][:200],
                }
                for r in selected
            ],
            total_latency_ms=round((time.time() - start) * 1000, 1),
        )
        print(
            f"[RAG] status=ok queries={log.n_queries} raw={log.n_candidates_raw} "
            f"unique={log.n_candidates_unique} selected={log.n_selected} "
            f"sources={log.source_distribution} latency={log.total_latency_ms}ms"
        )
        return context, log
    except Exception as e:
        print(f"[RAG] Retrieval failed (non-fatal): {e}")
        return "", _empty_log("error", n_q, per_query_k, final_k, mmr_lambda, start, str(e))
