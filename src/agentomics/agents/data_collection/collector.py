from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agentomics.agents.agent_utils import run_agent
from agentomics.agents.data_collection.capabilities import create_data_collection_capabilities
from agentomics.agents.data_collection.reviewer import review_collected_data
from agentomics.agents.data_collection.spec import DataCollectionSpec
from agentomics.runtime.collected_data_directory import (
    get_manifest_size_bytes,
    load_manifest,
    save_data_collection_output,
)
from agentomics.runtime.filesystem import ByteBudget
from agentomics.tools.discard_file_tool import create_discard_file_tool
from agentomics.tools.download_file_tool import DownloadDeps, create_download_file_tool
from agentomics.utils.data_module_config import DataModuleConfig

COLLECTOR_SYSTEM_PROMPT = f"""You collect public data for a machine-learning task described in the spec below.
Search the web and read pages to find data that fit the spec. Download the original files, and discard any \
downloaded file that turns out not to fit. The downloaded files are in {DataModuleConfig.FILES_DIRNAME}/, and \
{DataModuleConfig.MANIFEST_FILENAME} lists them with their sources; look inside them. Search, read, download and \
discard in any order and as often as you need.
Choose what to download as the spec's strategy says, and start from the spec's known_sources when it lists any.
When the spec sets max_download_gb, all files together must stay within that many GB.
Files that are already in {DataModuleConfig.FILES_DIRNAME}/ count against the size limit: keep the ones that fit the \
spec, discard the others and add what is missing.
Download the files as their sources publish them, with the documentation that describes them. Do not process, \
convert or label the data.
If a source cannot be reached or needs a login, skip it and say so. Do not claim that you found all the data that exist.
When you are done, give your summary."""


class DataCollectionOutput(BaseModel):
    summary: str = Field(description="What was collected and from which sources, and what you looked for but did not find")


@dataclass(kw_only=True)
class DataCollectionDeps(DownloadDeps):
    spec: DataCollectionSpec
    reviewer_model: Model


async def collect_data(
    config: DataModuleConfig, spec: DataCollectionSpec, collector_model: Model, reviewer_model: Model,
) -> None:
    deps = DataCollectionDeps(
        config=config,
        byte_budget=ByteBudget(
            max_bytes=config.max_download_bytes, used_bytes=get_manifest_size_bytes(load_manifest(config)),
        ),
        spec=spec,
        reviewer_model=reviewer_model,
    )
    _, output = await run_agent(
        agent=_create_collector_agent(config, collector_model),
        user_prompt=_get_collector_prompt(config, spec),
        max_steps=config.max_steps,
        message_history=None,
        deps=deps,
    )
    save_data_collection_output(config, output)


def _get_collector_prompt(config: DataModuleConfig, spec: DataCollectionSpec) -> str:
    manifest = load_manifest(config)
    return (
        f"Spec:\n{spec.model_dump_json(indent=2)}\n\n"
        f"{DataModuleConfig.FILES_DIRNAME}/ holds {len(manifest)} files that take {get_manifest_size_bytes(manifest)} bytes."
    )


def _create_collector_agent(config: DataModuleConfig, model: Model) -> Agent[DataCollectionDeps, DataCollectionOutput]:
    agent = Agent(
        model=model,
        system_prompt=COLLECTOR_SYSTEM_PROMPT,
        tools=[create_download_file_tool(config), create_discard_file_tool(config)],
        capabilities=create_data_collection_capabilities(config),
        model_settings=ModelSettings(temperature=config.temperature),
        output_type=DataCollectionOutput,
        retries={"output": config.max_validation_retries, "tools": config.max_tool_retries},
        deps_type=DataCollectionDeps,
    )

    @agent.output_validator
    async def _require_downloaded_files(ctx: RunContext[DataCollectionDeps], result: DataCollectionOutput) -> DataCollectionOutput:
        if not load_manifest(ctx.deps.config):
            raise ModelRetry("No files were downloaded.")
        return result

    @agent.output_validator
    async def _require_accepting_review(ctx: RunContext[DataCollectionDeps], result: DataCollectionOutput) -> DataCollectionOutput:
        review = await review_collected_data(ctx.deps.config, ctx.deps.spec, ctx.deps.reviewer_model, result.summary)
        if not review.accepted:
            raise ModelRetry(f"A reviewer did not accept your files:\n{review.model_dump_json(indent=2)}")
        return result

    return agent
