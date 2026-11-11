from __future__ import annotations

import json

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agentomics.agents.agent_utils import run_agent
from agentomics.agents.data_collection.capabilities import create_data_collection_capabilities
from agentomics.agents.data_collection.spec import DataCollectionSpec
from agentomics.runtime.collected_data_directory import (
    append_data_collection_review,
    get_manifest_size_bytes,
    load_manifest,
    read_manifest,
)
from agentomics.utils.data_module_config import DataModuleConfig


class DataCollectionReview(BaseModel):
    accepted: bool = Field(description="Whether the files fit the spec and no data were missed")
    issues: list[str] = Field(
        default=[],
        description=(
            "What to fix when the files are not accepted. One issue per source (a dataset, a repository or a website), "
            "not one per file of that source, each with the source or URL and what to do"
        ),
    )

    @model_validator(mode="after")
    def _check_issues(self) -> DataCollectionReview:
        if not self.accepted and not self.issues:
            raise ValueError("a review that does not accept the files lists at least one issue")
        return self


async def review_collected_data(
    config: DataModuleConfig, spec: DataCollectionSpec, reviewer_model: Model, summary: str,
) -> DataCollectionReview:
    _, review = await run_agent(
        agent=_create_reviewer_agent(config, reviewer_model),
        user_prompt=_get_reviewer_prompt(config, spec, summary),
        max_steps=config.max_steps,
        message_history=None,
    )
    append_data_collection_review(config, review)
    return review


def _get_reviewer_system_prompt(config: DataModuleConfig) -> str:
    return f"""You review the files another agent downloaded for a machine-learning task. You receive the spec of the \
task, the manifest of the downloaded files with their sources and their paths, the bytes they take with the size \
limit, and the agent's summary of the collection.
Look inside the downloaded files and check that each file fits the spec and that the files were chosen as the spec's \
strategy says.
Data were missed only in one of these cases:
- A source in the spec's known_sources was neither downloaded nor reported in the summary as unreachable.
- A downloaded file is a part or a processed version of data whose original or fuller version is published.
- A web search of your own finds a source that fits the spec and is neither downloaded nor named in the summary. \
Run at most {config.max_reviewer_searches} such searches.
The files stay as their sources publish them, except that archives are unpacked. Do not ask for processed or \
converted files. Do not ask for records fetched one at a time from an API when the source offers a bulk export.
Accept when the files fit the spec and no data were missed. Otherwise list each issue concretely, with \
the source or URL to use. Do not reject only because the available data are limited or the size limit leaves \
relevant data out. Do not ask again for files the summary reports as unreachable unless reading the page shows that \
they open now."""


def _get_reviewer_prompt(config: DataModuleConfig, spec: DataCollectionSpec, summary: str) -> str:
    byte_counts = {"used_bytes": get_manifest_size_bytes(load_manifest(config)), "max_bytes": config.max_download_bytes}
    return "\n\n".join([
        f"Spec:\n{spec.model_dump_json(indent=2)}",
        f"Manifest:\n{read_manifest(config)}",
        f"Bytes used and size limit:\n{json.dumps(byte_counts)}",
        f"Summary:\n{summary}",
    ])


def _create_reviewer_agent(config: DataModuleConfig, model: Model) -> Agent[None, DataCollectionReview]:
    return Agent(
        model=model,
        system_prompt=_get_reviewer_system_prompt(config),
        capabilities=create_data_collection_capabilities(config),
        model_settings=ModelSettings(temperature=config.temperature),
        output_type=DataCollectionReview,
        retries={"output": config.max_validation_retries, "tools": config.max_tool_retries},
    )
