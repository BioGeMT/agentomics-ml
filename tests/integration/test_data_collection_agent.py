import json

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentomics.agents.data_collection import DataRequest, build_collector, collect_data
from agentomics.tools.download_tool import DownloadFolder
from tests.helpers import serve_files

SUMMARY = "files/a.csv and files/b.csv hold the measurements."


REQUEST = DataRequest(
    task_type="classification",
    description="Predict a binary label.",
    inputs=["measurements"],
    label="diagnosis",
    classes=["benign", "malignant"],
)


def report(info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"summary": SUMMARY})])


def is_reviewer(info: AgentInfo) -> bool:
    return all(tool.name != "download_file" for tool in info.function_tools)


def review(info: AgentInfo, accepted: bool) -> ModelResponse:
    verdict = {"accepted": accepted, "issues": [] if accepted else ["a known source is missing"]}
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])


def test_collection_keeps_the_downloads_and_ends_when_a_review_accepts(tmp_path):
    verdicts = iter([False, True])
    with serve_files({"/a.csv": b"a" * 100, "/b.csv": b"b" * 100}) as base_url:
        def respond(messages, info: AgentInfo) -> ModelResponse:
            if is_reviewer(info):
                return review(info, accepted=next(verdicts))
            if len(messages) == 1:
                return ModelResponse(parts=[
                    ToolCallPart("download_file", {"url": f"{base_url}/a.csv", "filename": "a.csv"}),
                    ToolCallPart("download_file", {"url": f"{base_url}/b.csv", "filename": "b.csv"}),
                ])
            return report(info)

        assert collect_data(REQUEST, tmp_path, FunctionModel(respond)) == 0

    folder = DownloadFolder(tmp_path, None)
    assert sorted(entry.path for entry in folder.entries) == ["a.csv", "b.csv"]
    assert folder.check() == []
    assert [verdict["accepted"] for verdict in json.loads((tmp_path / "review.json").read_text())] == [False, True]
    assert (tmp_path / "summary.md").read_text() == SUMMARY


def test_collection_fails_after_four_rejected_reviews(tmp_path):
    folder = DownloadFolder(tmp_path, None)
    with serve_files({"/a.csv": b"a" * 100}) as base_url:
        folder.download_file(f"{base_url}/a.csv", "a.csv")

    def respond(messages, info: AgentInfo) -> ModelResponse:
        return review(info, accepted=False) if is_reviewer(info) else report(info)

    with pytest.raises(UnexpectedModelBehavior):
        build_collector(FunctionModel(respond), REQUEST, folder).run_sync("collect")

    assert len(json.loads((tmp_path / "review.json").read_text())) == 4
