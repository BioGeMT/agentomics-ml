from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool

from agentomics.agents.agent_utils import create_read_only_file_capabilities, run_agent
from agentomics.runtime.paper_retrieval import prepare_papers
from agentomics.utils.config import Config
from agentomics.utils.exceptions import AgentRunFailed

PAPER_QA_MAX_TOOL_CALLS = 20
PAPER_QA_MAX_CONTEXT_TOKENS = 100_000


class PaperSource(BaseModel):
    path: str = Field(description="Supporting file path relative to the processed papers directory.")
    line: int | None = Field(default=None, description="Supporting line number as printed by read_file; omit for a page image.")


class PaperAnswer(BaseModel):
    answer: str = Field(description="Concise answer based on the papers, acknowledging any missing evidence.")
    sources: list[PaperSource] = Field(description="Files and passages inspected that support the answer.")


def create_paper_qa_tool(config: Config) -> Tool[Any]:
    async def paper_qa(ctx: RunContext, question: str) -> PaperAnswer:
        """Read downloaded papers.

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
            capabilities=create_read_only_file_capabilities(root, PAPER_QA_MAX_TOOL_CALLS, PAPER_QA_MAX_CONTEXT_TOKENS),
            retries={"output": config.max_validation_retries, "tools": config.max_tool_retries},
        )
        try:
            _, answer = await run_agent(
                agent=reader, user_prompt=question, max_steps=2 * PAPER_QA_MAX_TOOL_CALLS, message_history=None,
            )
        except AgentRunFailed as error:
            raise ModelRetry("Paper reading failed; ask a narrower question or continue without it.") from error
        return answer

    return Tool(
        paper_qa,
        name="paper_qa",
        description=(
            "Ask a question about papers downloaded with paper_fetch. A separate reader searches the collection, "
            f"including figures, and returns an answer with sources relative to {config.processed_papers_dir} "
            "that you can inspect with Bash."
        ),
        takes_ctx=True,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        sequential=True,
    )
