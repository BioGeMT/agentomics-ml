import json
from collections.abc import Iterable

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentomics.agents.data_collection import DataRequest, collect_data
from agentomics.runtime.collected_data import load_manifest
from agentomics.tools.download_tool import DownloadDeps, check_collected_data, download_file
from agentomics.utils.exceptions import IterationRunFailed
from tests.helpers import serve_files

SUMMARY = "files/a.csv and files/b.csv hold the measurements."
REQUEST = DataRequest(
    task_type="classification",
    description="Predict a binary label.",
    inputs=["measurements"],
    label="diagnosis",
    classes=["benign", "malignant"],
)


def _model_downloading_two_files(base_url: str, review_verdicts: Iterable[bool]) -> FunctionModel:
    """Create a local model that downloads a.csv and b.csv as the collector and gives the verdicts as the reviewer."""
    verdicts = iter(review_verdicts)

    def respond(messages, info: AgentInfo) -> ModelResponse:
        output_tool = info.output_tools[0].name
        if all(tool.name != download_file.__name__ for tool in info.function_tools):
            accepted = next(verdicts)
            issues = [] if accepted else ["a known source is missing"]
            return ModelResponse(parts=[ToolCallPart(output_tool, {"accepted": accepted, "issues": issues})])
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(download_file.__name__, {"url": f"{base_url}/a.csv", "filename": "a.csv"}),
                ToolCallPart(download_file.__name__, {"url": f"{base_url}/b.csv", "filename": "b.csv"}),
            ])
        return ModelResponse(parts=[ToolCallPart(output_tool, {"summary": SUMMARY})])

    return FunctionModel(respond)


def test_collection_keeps_the_downloads_and_ends_when_a_review_accepts(tmp_path):
    with serve_files({"/a.csv": b"a" * 100, "/b.csv": b"b" * 100}) as base_url:
        exit_code = collect_data(REQUEST, tmp_path, _model_downloading_two_files(base_url, [False, True]))

    assert exit_code == 0
    assert sorted(entry.path for entry in load_manifest(tmp_path)) == ["files/a.csv", "files/b.csv"]
    assert check_collected_data(DownloadDeps(collected_data_directory=tmp_path, max_bytes=None), verify_hashes=True) == []
    assert [verdict["accepted"] for verdict in json.loads((tmp_path / "review.json").read_text())] == [False, True]
    assert (tmp_path / "summary.md").read_text() == SUMMARY


def test_collection_fails_after_four_rejected_reviews(tmp_path):
    with serve_files({"/a.csv": b"a" * 100, "/b.csv": b"b" * 100}) as base_url:
        with pytest.raises(IterationRunFailed):
            collect_data(REQUEST, tmp_path, _model_downloading_two_files(base_url, [False] * 4))

    assert len(json.loads((tmp_path / "review.json").read_text())) == 4
    assert not (tmp_path / "summary.md").exists()
