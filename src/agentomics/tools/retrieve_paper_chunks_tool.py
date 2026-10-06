from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import Tool

from agentomics.runtime.paper_retrieval import retrieve_paper_chunks
from agentomics.utils.config import Config


def create_retrieve_paper_chunks_tool(config: Config) -> Tool[Any]:
    def _retrieve_paper_chunks(
        query: Annotated[str, Field(min_length=1)],
        max_results: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> list[dict[str, str | int | float]]:
        """Find passages relevant to a query in the fetched scientific papers.

        Args:
            query: Information to find in the fetched papers.
            max_results: Maximum number of passages to return.
        """
        return retrieve_paper_chunks(config, query, max_results)

    return Tool(
        _retrieve_paper_chunks,
        name="retrieve_paper_chunks",
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        sequential=True,
    )
