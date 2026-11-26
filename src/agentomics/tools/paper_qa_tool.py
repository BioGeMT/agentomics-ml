from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool

from agentomics.agents.agent_utils import run_agent
from agentomics.agents.capabilities import create_read_only_file_capabilities
from agentomics.runtime.paper_retrieval import prepare_papers
from agentomics.utils.config import Config
from agentomics.utils.exceptions import AgentRunFailed


class PaperSource(BaseModel):
    path: str = Field(description="Supporting file path relative to the processed papers directory.")
    line: int | None = Field(default=None, description="Supporting line number as printed by read_file; omit for a page image.")


class PaperAnswer(BaseModel):
    answer: str = Field(description="Concise answer based on the papers, acknowledging any missing evidence.")
    sources: list[PaperSource] = Field(description="Only inspected files and passages supporting claims in the answer.")


def create_paper_qa_tool(config: Config) -> Tool[Any]:
    async def paper_qa(ctx: RunContext, question: str) -> PaperAnswer:
        """Ask about any content in papers downloaded with paper_fetch.

        A separate reader searches the collection and returns a concise answer with supporting sources.
        Ask a follow-up question when more detail is needed.

        Args:
            question: Self-contained question to answer using the downloaded papers.
        """
        root = prepare_papers(config)
        reader = Agent(
            ctx.model,
            name="paper_reader",
            output_type=PaperAnswer,
            instructions=(
                "Answer concisely using only the papers in this directory. Each document.md has '## Page N' "
                "sections linking to rendered page images. Search before reading, read relevant passages, "
                "and view page images when figures or tables matter. Cite only files you inspected, using "
                "paths relative to this directory and the printed line numbers. Omit line numbers for images. "
                "State when the papers do not answer the question or an image could not be inspected."
            ),
            capabilities=create_read_only_file_capabilities(
                config=config, root_dir=root,
                max_tool_calls=config.paper_qa_max_tool_calls,
                max_context_tokens=config.paper_qa_max_context_tokens,
            ),
            retries={"output": config.max_validation_retries, "tools": config.max_tool_retries},
        )
        try:
            _, answer = await run_agent(
                agent=reader, user_prompt=question, max_steps=2 * config.paper_qa_max_tool_calls, message_history=None,
            )
        except AgentRunFailed as error:
            raise ModelRetry("Paper reading failed; ask again, ask a narrower question, or continue without it.") from error
        return answer

    return Tool(
        paper_qa,
        name=Config.PAPER_QA_TOOL_NAME,
        takes_ctx=True,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        sequential=True,
    )
