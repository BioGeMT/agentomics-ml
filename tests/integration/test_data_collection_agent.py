import asyncio
import json
from collections.abc import Iterable
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentomics.agents.data_collection.collector import DataCollectionOutput, collect_data
from agentomics.agents.data_collection.reviewer import DataCollectionReview
from agentomics.agents.data_collection.spec import DataCollectionSpec
from agentomics.runtime.collected_data_directory import load_manifest
from agentomics.tools import web_fetch_tool
from agentomics.tools.web_fetch_tool import create_web_fetch_tool
from agentomics.utils.exceptions import IterationRunFailed
from tests.helpers import create_response_calling_output_tool, mock_downloads

SUMMARY = "Measurements from one source, with the documentation that describes them."
URL = "https://example.org"
SPEC = DataCollectionSpec(
    task_type="classification",
    description="Predict a binary label.",
    inputs=["measurements"],
    label="diagnosis",
    classes=["benign", "malignant"],
)


def _create_collector_model(responses: list[ModelResponse]) -> FunctionModel:
    """Create a model that gives the responses in turn and then its summary, as often as it is asked."""
    remaining_responses = iter(responses)

    def respond(_messages, info: AgentInfo) -> ModelResponse:
        response = next(remaining_responses, None)
        if response is None:
            return create_response_calling_output_tool(info, DataCollectionOutput(summary=SUMMARY))
        return response

    return FunctionModel(respond)


def _create_reviewer_model(review_verdicts: Iterable[bool]) -> FunctionModel:
    verdicts = iter(review_verdicts)

    def respond(_messages, info: AgentInfo) -> ModelResponse:
        accepted = next(verdicts)
        review = DataCollectionReview(accepted=accepted, issues=[] if accepted else ["a known source is missing"])
        return create_response_calling_output_tool(info, review)

    return FunctionModel(respond)


def _create_download_response(download_file_tool, *filenames: str) -> ModelResponse:
    return ModelResponse(parts=[
        ToolCallPart(download_file_tool.name, {"url": f"{URL}/{filename}", "filename": filename})
        for filename in filenames
    ])


def _load_review_verdicts(config) -> list[bool]:
    return [json.loads(line)["accepted"] for line in config.reviews_path.read_text(encoding="utf-8").splitlines()]


def test_collection_keeps_the_downloads_and_ends_when_a_review_accepts(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/a.csv": b"a" * 100, f"{URL}/b.csv": b"b" * 100})
    collector_model = _create_collector_model([_create_download_response(download_file_tool, "a.csv", "b.csv")])

    asyncio.run(collect_data(data_module_config, SPEC, collector_model, _create_reviewer_model([False, True])))

    assert [entry.path for entry in load_manifest(data_module_config)] == ["files/a.csv", "files/b.csv"]
    assert _load_review_verdicts(data_module_config) == [False, True]
    assert json.loads(data_module_config.output_path.read_text(encoding="utf-8")) == {"summary": SUMMARY}


def test_collection_fails_when_no_review_accepts_and_keeps_the_downloads(data_module_config, download_file_tool, monkeypatch):
    mock_downloads(monkeypatch, {f"{URL}/a.csv": b"a" * 100})
    collector_model = _create_collector_model([_create_download_response(download_file_tool, "a.csv")])
    review_count = data_module_config.max_validation_retries + 1

    with pytest.raises(IterationRunFailed):
        asyncio.run(collect_data(data_module_config, SPEC, collector_model, _create_reviewer_model([False] * review_count)))

    assert _load_review_verdicts(data_module_config) == [False] * review_count
    assert [entry.path for entry in load_manifest(data_module_config)] == ["files/a.csv"]
    assert not data_module_config.output_path.exists()


def test_summary_without_downloaded_files_is_not_reviewed(data_module_config):
    with pytest.raises(IterationRunFailed):
        asyncio.run(collect_data(data_module_config, SPEC, _create_collector_model([]), _create_reviewer_model([])))

    assert not data_module_config.reviews_path.exists()


def test_collection_continues_after_page_reads_fail_in_a_row(data_module_config, download_file_tool, monkeypatch):
    async def fail_to_fetch(url: str) -> None:
        raise ModelRetry(f"{url} could not be fetched")

    monkeypatch.setattr(web_fetch_tool, "web_fetch_tool", lambda: SimpleNamespace(function=fail_to_fetch))
    mock_downloads(monkeypatch, {f"{URL}/a.csv": b"a" * 100})
    web_fetch_tool_name = create_web_fetch_tool(data_module_config).name
    collector_model = _create_collector_model([
        ModelResponse(parts=[ToolCallPart(web_fetch_tool_name, {"url": f"{URL}/one"})]),
        ModelResponse(parts=[ToolCallPart(web_fetch_tool_name, {"url": f"{URL}/two"})]),
        _create_download_response(download_file_tool, "a.csv"),
    ])

    asyncio.run(collect_data(data_module_config, SPEC, collector_model, _create_reviewer_model([True])))

    assert [entry.path for entry in load_manifest(data_module_config)] == ["files/a.csv"]
