import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, Tool
from pydantic_ai.capabilities import LocalWorkspace, WebFetch, WebSearch
from pydantic_ai.common_tools.web_fetch import WebFetchResult, web_fetch_tool
from pydantic_ai.models import Model
from pydantic_ai_harness import FileSystem

from agentomics.agents.agent_utils import run_agent
from agentomics.runtime.collected_data import (
    FILES_DIRECTORY_NAME,
    MANIFEST_FILE_NAME,
    REVIEW_FILE_NAME,
    SUMMARY_FILE_NAME,
    find_problems,
    load_manifest,
    write_atomically,
)
from agentomics.tools.download_tool import (
    ARCHIVE_SUFFIXES,
    DownloadDeps,
    create_download_tools,
    current_budget,
    discard_file,
    download_file,
)
from agentomics.utils.config import Config
from agentomics.utils.exceptions import IterationRunFailed
from agentomics.utils.task_types import TaskTypes

BYTES_PER_GB = 1_000_000_000
MAX_REVIEWS = 4
# The pydantic-ai-harness file tools both agents get.
READ_FILE_TOOL = "read_file"
LIST_DIRECTORY_TOOL = "list_directory"

COLLECTOR_INSTRUCTIONS = f"""You collect public data for a machine-learning task described in the request below.
Search the web and read pages to find data that fit the request. Download the original files with \
{download_file.__name__}, and discard with {discard_file.__name__} any downloaded file that turns out not to fit. \
{download_file.__name__} unpacks archives ({', '.join(ARCHIVE_SUFFIXES)}) and removes the archive. The downloaded \
files are in {FILES_DIRECTORY_NAME}/; look inside them with {LIST_DIRECTORY_TOOL} and {READ_FILE_TOOL}. Search, read, \
download and discard in any order and as often as you need.
Choose what to download as the request's strategy says, and start from the request's known_sources when it lists any.
When the request sets max_download_gb, all files together must stay within that many GB; {download_file.__name__} \
reports the bytes used and the size limit.
A manifest below the request lists files that an earlier run left in {FILES_DIRECTORY_NAME}/. They count against the \
size limit: keep the ones that fit the request, discard the others and add what is missing.
Download the files as their sources publish them, with the documentation that describes them. Do not process, \
convert or label the data.
If a source cannot be reached or needs a login, skip it and say so. Do not claim that you found all the data that exist.
When you are done, give your summary."""

REVIEWER_INSTRUCTIONS = f"""You review the files another agent downloaded for a machine-learning task. You receive the \
request, the manifest of the downloaded files with their sources and their paths, the bytes they take with the size \
limit, and the agent's summary of them.
Use web search, page reading, and {LIST_DIRECTORY_TOOL} and {READ_FILE_TOOL} on the downloaded files to check that \
each file is what the summary says and fits the request, whether relevant public data were missed, including the \
user's known sources, and whether the files were chosen as the request's strategy says.
The files stay as their sources publish them, except that archives are unpacked. Do not ask for processed or \
converted files. Do not ask for records fetched one at a time from an API when the source offers a bulk export.
Accept when the files fit the request and nothing important is missing. Otherwise list each issue concretely, with \
the source or URL to use. Do not reject only because the available data are limited or the size limit leaves \
relevant data out. Do not ask again for files the summary reports as unreachable unless page reading shows that \
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


class CollectionSummary(BaseModel):
    summary: str = Field(description="What each file contains and where it comes from, and what you looked for but did not find")


class Review(BaseModel):
    accepted: bool = Field(description="Whether the files fit the request and nothing important is missing")
    issues: list[str] = Field(default=[], description="What to fix when the files are not accepted, one issue per item")

    @model_validator(mode="after")
    def _check_issues(self) -> "Review":
        if not self.accepted and not self.issues:
            raise ValueError("a review that does not accept the files lists at least one issue")
        return self


@dataclass(kw_only=True)
class CollectionDeps(DownloadDeps):
    request: DataRequest
    reviewer: Agent[None, Review]
    reviews: list[Review] = field(default_factory=list)


_REVIEWS = TypeAdapter(list[Review])


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
                f"Download it with {download_file.__name__} and look inside with {READ_FILE_TOOL}.")
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
        FileSystem(read_only=True, tools=[READ_FILE_TOOL, LIST_DIRECTORY_TOOL]),
    ]


def _build_reviewer(model: Model, collected_data_directory: Path) -> Agent[None, Review]:
    return Agent(
        model,
        instructions=REVIEWER_INSTRUCTIONS,
        output_type=Review,
        capabilities=_capabilities(collected_data_directory),
        retries={"tools": Config.DEFAULT_MAX_TOOL_RETRIES},
    )


def _build_collector(model: Model, collected_data_directory: Path) -> Agent[CollectionDeps, CollectionSummary]:
    collector = Agent(
        model,
        deps_type=CollectionDeps,
        instructions=COLLECTOR_INSTRUCTIONS,
        tools=create_download_tools(),
        capabilities=_capabilities(collected_data_directory),
        output_type=CollectionSummary,
        retries={"output": MAX_REVIEWS - 1},
    )
    collector.output_validator(_review_collection)
    return collector


async def _review_collection(ctx: RunContext[CollectionDeps], collection: CollectionSummary) -> CollectionSummary:
    deps = ctx.deps
    if not load_manifest(deps.collected_data_directory):
        raise ModelRetry("No files were downloaded.")
    review_prompt = "\n\n".join([
        f"Request:\n{deps.request.model_dump_json(indent=2)}",
        f"Manifest:\n{(deps.collected_data_directory / MANIFEST_FILE_NAME).read_text()}",
        f"Bytes used and size limit:\n{current_budget(deps).model_dump_json()}",
        f"Summary:\n{collection.summary}",
    ])
    review = (await deps.reviewer.run(review_prompt)).output
    deps.reviews.append(review)
    write_atomically(deps.collected_data_directory / REVIEW_FILE_NAME, _REVIEWS.dump_json(deps.reviews, indent=2))
    if not review.accepted:
        raise ModelRetry("A reviewer checked your files and found these issues:\n" + "\n".join(review.issues))
    return collection


def collect_data(
    request: DataRequest, collected_data_directory: Path, model: Model, allow_local_addresses: bool = False,
) -> int:
    max_bytes = None if request.max_download_gb is None else int(request.max_download_gb * BYTES_PER_GB)
    problems = find_problems(collected_data_directory, max_bytes, verify_hashes=False)
    if problems:
        print("The collected data folder cannot be used:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    deps = CollectionDeps(
        collected_data_directory=collected_data_directory,
        max_bytes=max_bytes,
        allow_local_addresses=allow_local_addresses,
        request=request,
        reviewer=_build_reviewer(model, collected_data_directory),
    )
    try:
        _, collection = asyncio.run(run_agent(
            _build_collector(model, collected_data_directory),
            _collector_prompt(request, collected_data_directory),
            max_steps=Config.DEFAULT_MAX_STEPS,
            message_history=None,
            deps=deps,
        ))
    except IterationRunFailed:
        message = "Data collection did not finish."
        if deps.reviews and not deps.reviews[-1].accepted:
            message += f" The reviewer did not accept the files; {REVIEW_FILE_NAME} lists the issues."
        print(
            f"{message} The downloaded files stay in the collected data folder; run the command again on it to continue.",
            file=sys.stderr,
        )
        return 1
    problems = find_problems(collected_data_directory, max_bytes, verify_hashes=True)
    if problems:
        print("The collected data folder does not match its manifest:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    write_atomically(collected_data_directory / SUMMARY_FILE_NAME, collection.summary.encode())
    return 0
