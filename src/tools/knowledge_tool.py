from pydantic_ai import Tool
import weave


def create_knowledge_integration_tool(max_retries: int):
    """On-demand RAG retrieval tool for `rag_od` knowledge mode.

    The agent calls this tool with a single natural-language query whenever
    it needs domain knowledge. Each call embeds the query, retrieves from
    pgvector, MMR-reranks, and returns a formatted markdown block of excerpts.

    Telemetry: each invocation is traced by weave via `_integrate_knowledge`.
    No file-log is written (by design — per-iteration tracking is handled via
    weave traces, and the tool has no clean `iteration` handle at call time).

    The import of retrieval helpers is deferred to factory call time so that
    non-RAG runs don't crash if psycopg2/requests are missing.
    """
    from rag.rag_utils.retrieval import run_rag_retrieval

    @weave.op(call_display_name="knowledge_integration")
    def _integrate_knowledge(query: str) -> str:
        """
        Retrieve domain knowledge from the curated scientific literature
        indexed in the RAG vector database. Use this whenever you need
        background information, methodological guidance, or dataset-specific
        insights to inform your decisions.

        Examples:
            "How to represent miRNA-target interaction sequences for deep learning"
            "Common evaluation pitfalls in CLASH datasets"
            "Feature engineering techniques for RNA binding prediction"

        Args:
            query: A single natural-language query describing the information
                you need. Call the tool multiple times with different queries
                if you need multiple pieces of information.

        Returns:
            A markdown-formatted block of the most relevant excerpts from the
            knowledge base, or an empty string if retrieval failed or the DB
            is empty.
        """
        context, _log = run_rag_retrieval(
            queries=[query],
            per_query_k=6,
            final_k=7,
            mmr_lambda=0.6,
        )
        if not context:
            return "No relevant knowledge found for this query."
        return context

    return Tool(
        function=_integrate_knowledge,
        takes_ctx=False,
        max_retries=max_retries,
        require_parameter_descriptions=True,
        name="knowledge_integration",
    )
