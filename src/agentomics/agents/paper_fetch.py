from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

from agentomics.tools.save_paper_pdf_tool import create_save_paper_pdf_tool
from agentomics.utils.config import Config


PAPER_FETCH_SYSTEM_PROMPT = """
Your goal is to find scientific papers informing a machine learning modeling task.

Work in three phases:
1. Check the returned search results for relevance to the query. If the same paper appears several times, choose a single host to retrieve it from.
2. Read. Use web_fetch on the web pages the search returned to confirm what each paper actually
   is and to find its PDF link. Publisher landing pages state the real PDF URL; read it rather than
   assembling one from the article id. Never report a paper you have not fetched.
3. Retrieve. Call save_paper_pdf for each paper relevant to the query. It writes the PDF to disk; you do not
   need to read the PDF contents yourself. If a URL returns a download page instead of a PDF, go back
   to the landing page for the direct link rather than guessing.

Report only papers whose PDF you successfully saved.
"""

class Paper(BaseModel):
    title: str = Field(
        description="The title of the paper."
        )
    pdf_url: str | None = Field(
        default=None,
        description="The URL the PDF was downloaded from.")
    pdf_filename: str | None = Field(
        default=None,
        description="Filename of the saved PDF, as returned by save_paper_pdf. "
        "Empty if the PDF could not be retrieved.")
    relevance: str = Field(
        description="Why this paper is relevant to the query, and what it contributes to the modeling task.")

class PaperFetchOutput(BaseModel):
    papers: list[Paper] = Field(
        description="List of relevant fetched papers. Empty if no relevant papers were retrieved or saved.")

def create_paper_fetch_agent(config: Config, model: Model) -> Agent[None, PaperFetchOutput]:
    agent = Agent(
        model=model,
        system_prompt=PAPER_FETCH_SYSTEM_PROMPT,
        tools=[
            web_fetch_tool(timeout=config.web_fetch_timeout),
            create_save_paper_pdf_tool(config),
        ],
        model_settings=ModelSettings(temperature=config.temperature),
        output_type=PaperFetchOutput,
        retries=config.max_validation_retries
    )
    attach_output_validator(agent, config)
    return agent

def attach_output_validator(agent: Agent[None, PaperFetchOutput], config: Config) -> None:
    @agent.output_validator
    async def _papers_stored_in_shared_dir(result: PaperFetchOutput) -> PaperFetchOutput:
        papers_dir = config.fetched_papers_dir.resolve()

        for paper in result.papers:
            if paper.pdf_filename is None:
                continue
            pdf_path = papers_dir / Path(paper.pdf_filename).name
            if not pdf_path.is_file():
                raise ModelRetry(f"'{paper.pdf_filename}' is not in {papers_dir}. Save it with save_paper_pdf or clear pdf_filename.")
        return result

def build_prompt(config: Config, query: str, websearch_result: list[dict]) -> str:
    return f"""Based on the query received as input:
    {query}
    and the results of web search for that query:
    {websearch_result}
    - Save the pdf of relevant papers at: {config.fetched_papers_dir}"""
