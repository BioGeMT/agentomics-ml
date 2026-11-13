from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agentomics.agents.data_collection.capabilities import create_data_collection_capabilities
from agentomics.agents.data_collection.reviewer import review_collected_data
from agentomics.agents.data_collection.spec import DataCollectionSpec
from agentomics.agents.steps.base import AgenticStepMixin, AgenticStepOutput, RuntimeStep
from agentomics.runtime.collected_data_directory import get_manifest_size_bytes, load_manifest
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


class DataCollectionOutput(AgenticStepOutput):
    summary: str = Field(description="What was collected and from which sources, and what you looked for but did not find")


@dataclass(kw_only=True)
class DataCollectionDeps(DownloadDeps):
    spec: DataCollectionSpec
    reviewer_model: Model


class DataCollectionStep(RuntimeStep, AgenticStepMixin):
    step_id = "data_collection"
    display_name = "DATA COLLECTION"
    output_type = DataCollectionOutput

    def __init__(
        self,
        config: DataModuleConfig,
        spec: DataCollectionSpec,
        collector_model: Model,
        reviewer_model: Model,
    ) -> None:
        super().__init__()
        self.config = config
        self.spec = spec
        self.collector_model = collector_model
        self.reviewer_model = reviewer_model

    @property
    def step_dir(self) -> Path:
        return self.config.collected_data_dir

    def step_validators(self) -> list[Callable]:
        async def _require_downloaded_files(ctx: RunContext[DataCollectionDeps], result: DataCollectionOutput) -> DataCollectionOutput:
            if not load_manifest(ctx.deps.config):
                raise ModelRetry("No files were downloaded.")
            return result

        async def _require_accepting_review(ctx: RunContext[DataCollectionDeps], result: DataCollectionOutput) -> DataCollectionOutput:
            review = await review_collected_data(ctx.deps.config, ctx.deps.spec, ctx.deps.reviewer_model, result.summary)
            if not review.accepted:
                raise ModelRetry(f"A reviewer did not accept your files:\n{review.model_dump_json(indent=2)}")
            return result
        return [_require_downloaded_files, _require_accepting_review]

    def build_deps(self) -> DataCollectionDeps:
        return DataCollectionDeps(
            config=self.config,
            byte_budget=ByteBudget(
                max_bytes=self.config.max_download_bytes, used_bytes=get_manifest_size_bytes(load_manifest(self.config)),
            ),
            spec=self.spec,
            reviewer_model=self.reviewer_model,
        )

    def build_user_prompt(self) -> str:
        manifest = load_manifest(self.config)
        return (
            f"Spec:\n{self.spec.model_dump_json(indent=2)}\n\n"
            f"{DataModuleConfig.FILES_DIRNAME}/ holds {len(manifest)} files that take {get_manifest_size_bytes(manifest)} bytes."
        )

    async def _execute(self) -> AgenticStepOutput:
        return await self.run_agent_step(
            user_prompt=self.build_user_prompt(),
            model=self.collector_model,
            model_settings=ModelSettings(temperature=self.config.temperature),
            max_steps=self.config.max_steps,
            max_validation_retries=self.config.max_validation_retries,
            max_tool_retries=self.config.max_tool_retries,
            tools=[create_download_file_tool(self.config), create_discard_file_tool(self.config)],
            capabilities=create_data_collection_capabilities(self.config),
            deps_type=DataCollectionDeps,
            system_prompt=COLLECTOR_SYSTEM_PROMPT,
        )
