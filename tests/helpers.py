import io
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agentomics.runtime.collected_data_directory import get_manifest_size_bytes, load_manifest
from agentomics.runtime.filesystem import ByteBudget
from agentomics.tools import download_file_tool
from agentomics.tools.download_file_tool import DownloadDeps
from agentomics.utils.config import Config
from agentomics.utils.data_module_config import DataModuleConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PR_TITLE_VALIDATOR = REPO_ROOT / "scripts" / "validate_pr_title.py"


def run_pr_title_validator(title: str) -> subprocess.CompletedProcess[str]:
    """Run the repository's PR-title policy validator."""
    return subprocess.run(
        [sys.executable, str(PR_TITLE_VALIDATOR), title],
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
        check=False,
    )


def run_git_cli_command(
    repo_dir: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """Run a Git CLI command in an isolated test repository."""
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def create_runtime_step(step_type, config: Config, *, model, trace_analysis_model=None):
    """Construct a runtime step with unused external dependencies replaced."""
    return step_type(
        config,
        model,
        Mock(),
        trace_analysis_model or Mock(),
        Mock(),
        [],
    )


def create_response_calling_output_tool(agent_info: AgentInfo, output: BaseModel) -> ModelResponse:
    """Create a model response that calls the agent's sole structured-output tool."""
    assert len(agent_info.output_tools) == 1
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=agent_info.output_tools[0].name,
                args=output.model_dump(),
            )
        ]
    )


def create_model_calling_output_tool(output: BaseModel) -> FunctionModel:
    """Create a local model that calls the agent's sole structured-output tool."""
    def respond(_messages, agent_info):
        return create_response_calling_output_tool(agent_info, output)

    return FunctionModel(respond)


def create_failing_model() -> FunctionModel:
    """Create a local model whose every request fails."""
    def respond(_messages, _agent_info):
        raise RuntimeError("runtime error")

    return FunctionModel(respond)


def write_step_file(config: Config, filename: str, content: str) -> Path:
    """Write a text file in the current step workspace."""
    path = config.current_step_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_split_folder(split_path: Path, row_id: str = "row-1") -> None:
    """Create a minimal labeled folder split."""
    input_dir = split_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "examples.txt").write_text(f"{row_id}\n", encoding="utf-8")
    (split_path / "labels.csv").write_text(
        f"id,numeric_label\n{row_id},0\n",
        encoding="utf-8",
    )


def create_zip_archive(members: list[tuple[str, bytes]]) -> bytes:
    """Create a zip archive that holds the members, given as file name and content."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members:
            bundle.writestr(name, data)
    return buffer.getvalue()


def list_files(root: Path) -> list[str]:
    """List the files below root, as sorted paths relative to it."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def create_download_context(config: DataModuleConfig, max_bytes: int | None = None) -> RunContext[DownloadDeps]:
    """Create the context a download tool call receives, with the byte budget of the collected data folder."""
    byte_budget = ByteBudget(max_bytes=max_bytes, used_bytes=get_manifest_size_bytes(load_manifest(config)))
    return RunContext(deps=DownloadDeps(config=config, byte_budget=byte_budget), model=TestModel(), usage=RunUsage())


def mock_downloads(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes], send_content_length: bool = True) -> None:
    """Make the download tool receive the files, given by URL, instead of sending HTTP requests."""
    def respond(request: httpx2.Request) -> httpx2.Response:
        body = files.get(str(request.url))
        if body is None:
            return httpx2.Response(404)
        content = body if send_content_length else iter([body])
        return httpx2.Response(200, headers={"content-type": "text/csv"}, content=content)

    original_client = httpx2.Client
    monkeypatch.setattr(
        download_file_tool.httpx2, "Client",
        lambda **kwargs: original_client(transport=httpx2.MockTransport(respond), **kwargs),
    )
