import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentomics.agents import data_collection
from agentomics.agents.data_collection import MAX_REVIEWS, CollectionSummary, DataRequest, Review, collect_data
from agentomics.runtime.collected_data import load_manifest
from agentomics.tools.download_tool import download_file
from tests.helpers import call_output_tool, create_failing_model, serve_files

SUMMARY = "files/a.csv and files/b.csv hold the measurements."
REQUEST = DataRequest(
    task_type="classification",
    description="Predict a binary label.",
    inputs=["measurements"],
    label="diagnosis",
    classes=["benign", "malignant"],
)


def _is_reviewer(info: AgentInfo) -> bool:
    return all(tool.name != download_file.__name__ for tool in info.function_tools)


def _review(info: AgentInfo, accepted: bool) -> ModelResponse:
    return call_output_tool(info, Review(accepted=accepted, issues=[] if accepted else ["a known source is missing"]))


def _download(base_url: str, *filenames: str) -> ModelResponse:
    return ModelResponse(parts=[
        ToolCallPart(download_file.__name__, {"url": f"{base_url}/{filename}", "filename": filename})
        for filename in filenames
    ])


def _collect(collected_data_directory: Path, base_url: str, review_verdicts: Iterable[bool]) -> int:
    """Run a collection in which the collector downloads a.csv and b.csv and the reviewer gives the verdicts."""
    verdicts = iter(review_verdicts)

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if _is_reviewer(info):
            return _review(info, accepted=next(verdicts))
        if len(messages) == 1:
            return _download(base_url, "a.csv", "b.csv")
        return call_output_tool(info, CollectionSummary(summary=SUMMARY))

    return collect_data(REQUEST, collected_data_directory, FunctionModel(respond), allow_local_addresses=True)


def test_collection_keeps_the_downloads_and_ends_when_a_review_accepts(tmp_path):
    with serve_files({"/a.csv": b"a" * 100, "/b.csv": b"b" * 100}) as base_url:
        exit_code = _collect(tmp_path, base_url, [False, True])

    assert exit_code == 0
    assert sorted(entry.path for entry in load_manifest(tmp_path)) == ["files/a.csv", "files/b.csv"]
    assert [verdict["accepted"] for verdict in json.loads((tmp_path / "review.json").read_text())] == [False, True]
    assert (tmp_path / "summary.md").read_text() == SUMMARY


def test_collection_fails_when_no_review_accepts_and_keeps_the_downloads(tmp_path, capsys):
    with serve_files({"/a.csv": b"a" * 100, "/b.csv": b"b" * 100}) as base_url:
        exit_code = _collect(tmp_path, base_url, [False] * MAX_REVIEWS)

    assert exit_code == 1
    assert "The reviewer did not accept the files" in capsys.readouterr().err
    assert len(json.loads((tmp_path / "review.json").read_text())) == MAX_REVIEWS
    assert sorted(entry.path for entry in load_manifest(tmp_path)) == ["files/a.csv", "files/b.csv"]
    assert not (tmp_path / "summary.md").exists()


def test_reviews_file_holds_the_reviews_of_the_latest_run(tmp_path):
    with serve_files({"/a.csv": b"a" * 100, "/b.csv": b"b" * 100}) as base_url:
        _collect(tmp_path, base_url, [False, True])
        _collect(tmp_path, base_url, [True])

    assert [verdict["accepted"] for verdict in json.loads((tmp_path / "review.json").read_text())] == [True]


def test_collection_does_not_start_on_a_folder_that_disagrees_with_its_manifest(tmp_path, capsys):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "unrecorded.csv").write_bytes(b"x")

    exit_code = collect_data(REQUEST, tmp_path, create_failing_model())

    assert exit_code == 1
    assert "files/unrecorded.csv is not in the manifest" in capsys.readouterr().err


def test_page_reading_points_to_the_download_tool_for_a_file_the_model_cannot_take(tmp_path, monkeypatch):
    async def fetch_archive(url: str) -> BinaryContent:
        return BinaryContent(data=b"PK", media_type="application/zip")

    monkeypatch.setattr(data_collection, "web_fetch_tool", lambda: SimpleNamespace(function=fetch_archive))
    page_reading_results = []

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if _is_reviewer(info):
            return _review(info, accepted=True)
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("web_fetch", {"url": "https://example.org/data.zip"})])
        if not page_reading_results:
            page_reading_results.append(messages[-1].parts[0].content)
            return _download(base_url, "a.csv")
        return call_output_tool(info, CollectionSummary(summary=SUMMARY))

    with serve_files({"/a.csv": b"a" * 100}) as base_url:
        exit_code = collect_data(REQUEST, tmp_path, FunctionModel(respond), allow_local_addresses=True)

    assert exit_code == 0
    assert download_file.__name__ in page_reading_results[0]
