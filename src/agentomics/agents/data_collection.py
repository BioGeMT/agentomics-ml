import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, Tool
from pydantic_ai.capabilities import LocalWorkspace, WebFetch, WebSearch
from pydantic_ai.common_tools.web_fetch import WebFetchResult, web_fetch_tool
from pydantic_ai.models import Model
from pydantic_ai_harness import FileSystem

from agentomics.agents.agent_utils import run_agent
from agentomics.runtime.collected_data import MANIFEST_FILE_NAME, load_manifest
from agentomics.runtime.filesystem import restore_host_ownership
from agentomics.runtime.run_workflow import prepare_codex_credentials
from agentomics.tools.download_tool import DownloadDeps, check_collected_data, create_download_tools, current_budget
from agentomics.utils.config import Config
from agentomics.utils.providers.provider import Provider, get_provider_and_api_key
from agentomics.utils.task_types import TaskTypes

SUMMARY_FILE_NAME = "summary.md"
REVIEW_FILE_NAME = "review.json"
BYTES_PER_GB = 1_000_000_000

COLLECTOR_INSTRUCTIONS = """You collect public data for a machine-learning task described in the request below.
Search the web and read pages to find data that fit the request. Download the original files with download_file, \
and discard with discard_file any downloaded file that turns out not to fit. download_file unpacks zip, tar and \
gz archives into files/ and removes the archive. Look inside downloaded files with list_directory and read_file. \
Search, read, download and discard in any order and as often as you need.
Choose what to download as the request's strategy says, and start from the request's known_sources when it lists any.
When the request sets max_download_gb, all files together must stay within that many GB; download_file reports the \
bytes used and the size limit.
A manifest below the request lists files that an earlier run left in files/. They count against the size limit: keep \
the ones that fit the request, discard the others and add what is missing.
Download the files as their sources publish them, with the documentation that describes them. Do not process, \
convert or label the data.
If a source cannot be reached or needs a login, skip it and say so. Do not claim that you found all the data that exist.
When you are done, give your final answer: what each file in the folder contains and where it comes from, and \
what you looked for but did not find."""

REVIEWER_INSTRUCTIONS = """You review the files another agent downloaded for a machine-learning task. You receive the \
request, the manifest of downloaded files with their sources and their paths inside files/, the bytes they take \
with the size limit, and the agent's account of them.
Use web search, page reading, and list_directory and read_file on the downloaded files to check that each file is \
what the account says and fits the request, whether relevant public data were missed, including the user's known \
sources, and whether the files were chosen as the request's strategy says.
The files stay as their sources publish them, except that archives are unpacked. Do not ask for processed or \
converted files. Do not ask for records fetched one at a time from an API when the source offers a bulk export.
Accept when the files fit the request and nothing important is missing. Otherwise list each issue concretely, with \
the source or URL to use. Do not reject only because the available data are limited or the size limit leaves \
relevant data out. Do not ask again for files the account reports as unreachable unless page reading shows that \
they open now."""

DEFAULT_STRATEGY = "Download the most relevant, highest-quality sources first, then the rest of the relevant data."


class DataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskTypes
    description: str = Field(min_length=1)
    inputs: list[str] = Field(min_length=1)
    label: str = Field(min_length=1)
    classes: list[str] | None = None
    known_sources: list[str] = []
    max_download_gb: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    strategy: str = Field(default=DEFAULT_STRATEGY, min_length=1)

    @model_validator(mode="after")
    def _check_classes(self) -> "DataRequest":
        if self.task_type == TaskTypes.CLASSIFICATION and len(set(self.classes or [])) < 2:
            raise ValueError("classification needs at least two distinct classes")
        if self.task_type == TaskTypes.REGRESSION and self.classes:
            raise ValueError("regression takes no classes")
        return self


class CollectionReport(BaseModel):
    summary: str = Field(description="What each file contains and where it comes from, and what you looked for but did not find")


class Review(BaseModel):
    accepted: bool
    issues: list[str] = []


@dataclass(kw_only=True)
class CollectionDeps(DownloadDeps):
    request: DataRequest
    reviewer: Agent[None, Review]


async def _web_fetch(url: str) -> WebFetchResult | BinaryContent | str:
    """Fetches the content of a web page at the given URL and returns it as markdown or binary content.

    Args:
        url: The URL to fetch.
    """
    result = await web_fetch_tool().function(url)
    # pydantic-ai passes a fetched non-text file to the model as is, and its OpenAI models raise RuntimeError on
    # most types other than images and PDFs, which ends the run: https://github.com/pydantic/pydantic-ai/issues/11829
    if isinstance(result, BinaryContent) and not (result.is_image or result.media_type == "application/pdf"):
        return (f"{url} is a {result.media_type} file of {len(result.data)} bytes, not a page. "
                "Download it with download_file and look inside with read_file.")
    return result


def _collector_prompt(request: DataRequest, collected_data_directory: Path) -> str:
    sections = [f"Request:\n{request.model_dump_json(indent=2)}"]
    manifest_path = collected_data_directory / MANIFEST_FILE_NAME
    if manifest_path.exists():
        sections.append(f"Manifest of the files an earlier run left:\n{manifest_path.read_text()}")
    summary_path = collected_data_directory / SUMMARY_FILE_NAME
    if summary_path.exists():
        sections.append(f"Summary of the last accepted run:\n{summary_path.read_text()}")
    return "\n\n".join(sections)


def _capabilities(collected_data_directory: Path) -> list:
    return [
        WebSearch(local=False),
        # TODO: drop _web_fetch and use pydantic-ai's own web fetch once
        # https://github.com/pydantic/pydantic-ai/issues/11829 is fixed.
        WebFetch(local=Tool(_web_fetch, name="web_fetch")),
        LocalWorkspace(collected_data_directory, read_only=True),
        FileSystem(read_only=True, tools=["read_file", "list_directory"]),
    ]


def build_reviewer(model: Model, collected_data_directory: Path) -> Agent[None, Review]:
    return Agent(
        model,
        instructions=REVIEWER_INSTRUCTIONS,
        output_type=Review,
        capabilities=_capabilities(collected_data_directory),
        retries={"tools": Config.DEFAULT_MAX_TOOL_RETRIES},
    )


def build_collector(model: Model, collected_data_directory: Path) -> Agent[CollectionDeps, CollectionReport]:
    collector = Agent(
        model,
        deps_type=CollectionDeps,
        instructions=COLLECTOR_INSTRUCTIONS,
        tools=create_download_tools(),
        capabilities=_capabilities(collected_data_directory),
        output_type=CollectionReport,
        # The first answer plus three corrections: at most four reviews.
        retries={"output": 3},
    )
    collector.output_validator(review_collection)
    return collector


async def review_collection(ctx: RunContext[CollectionDeps], report: CollectionReport) -> CollectionReport:
    deps = ctx.deps
    if not load_manifest(deps.collected_data_directory):
        raise ModelRetry("No files were downloaded.")
    review_prompt = "\n\n".join([
        f"Request:\n{deps.request.model_dump_json(indent=2)}",
        f"Manifest:\n{(deps.collected_data_directory / MANIFEST_FILE_NAME).read_text()}",
        f"Bytes used and size limit:\n{current_budget(deps).model_dump_json()}",
        f"The agent's account:\n{report.summary}",
    ])
    review = (await deps.reviewer.run(review_prompt)).output
    review_path = deps.collected_data_directory / REVIEW_FILE_NAME
    reviews = json.loads(review_path.read_text()) if review_path.exists() else []
    reviews.append(review.model_dump())
    review_path.write_text(json.dumps(reviews, indent=2))
    if not review.accepted:
        raise ModelRetry("A reviewer checked your files and found these issues:\n" + "\n".join(review.issues))
    return report


def collect_data(request: DataRequest, collected_data_directory: Path, model: Model) -> int:
    max_bytes = None if request.max_download_gb is None else int(request.max_download_gb * BYTES_PER_GB)
    deps = CollectionDeps(
        collected_data_directory=collected_data_directory,
        max_bytes=max_bytes,
        request=request,
        reviewer=build_reviewer(model, collected_data_directory),
    )
    problems = check_collected_data(deps, verify_hashes=False)
    if problems:
        print("The collected data folder cannot be used:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    _, report = asyncio.run(run_agent(
        build_collector(model, collected_data_directory),
        _collector_prompt(request, collected_data_directory),
        max_steps=Config.DEFAULT_MAX_STEPS,
        message_history=None,
        deps=deps,
    ))
    problems = check_collected_data(deps, verify_hashes=True)
    if problems:
        print("The collected data folder does not match its manifest:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    (collected_data_directory / SUMMARY_FILE_NAME).write_text(report.summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the data collection agent inside the Agentomics image.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--collected-data-dir", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args(argv)

    request = DataRequest.model_validate_json(arguments.request.read_text())
    prepare_codex_credentials()
    api_key, provider_name = get_provider_and_api_key(arguments.provider)
    provider = Provider.create_provider(provider_name, api_key)
    model = provider.create_model_from_settings(
        arguments.model, Config.DEFAULT_USE_PROXY, Config.DEFAULT_LLM_RESPONSE_TIMEOUT,
    )
    try:
        return collect_data(request, arguments.collected_data_dir, model)
    finally:
        restore_host_ownership(arguments.collected_data_dir)


if __name__ == "__main__":
    sys.exit(main())
