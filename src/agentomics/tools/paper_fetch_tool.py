from typing import Any

from pydantic_ai import ModelRetry, RunContext, Tool
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from ddgs.exceptions import DDGSException

from agentomics.agents.paper_fetch import create_paper_fetch_agent, PaperFetchOutput, build_prompt
from agentomics.utils.config import Config


def create_paper_fetch_tool(config: Config, model: Model) -> Tool[Any]:
    async def paper_fetch(ctx: RunContext[dict], query: str) -> PaperFetchOutput:
        """
        A tool used to fetch scientific papers relevant to the provided query.
        The fetched papers are saved in PDF format in the fetched_papers/ , inside the shared folder .
        This tool should be called with one query at a time; call this tool multiple times if multiple topics need to be searched.

        Args:
            query: A string describing one scientific topic or question for which relevant papers are to be fetched.
        """
        try:
            websearch_result = await duckduckgo_search_tool(max_results=config.paper_fetch_max_results).function(query=query)
        except DDGSException as e:
            raise ModelRetry(f"Web search is currently unavailable: {e}. Retry paper_fetch or continue without literature.") from e
        agent = create_paper_fetch_agent(config, model)
        try:
            result = await agent.run(
                user_prompt=build_prompt(config, query, websearch_result),
                usage_limits=UsageLimits(request_limit=config.paper_fetch_request_limit),
            )
        except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
            raise ModelRetry(f"paper_fetch tool could not complete the search: {e}. Continue without it or retry with a different query.") from e
        return result.output

    return Tool(paper_fetch, name="paper_fetch")
