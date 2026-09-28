import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.utils.config import Config

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


def create_runtime_step(step_type, config: Config, *, model):
    """Construct a runtime step with unused external dependencies replaced."""
    return step_type(
        config,
        model,
        Mock(),
        Mock(),
        [],
    )


def create_model_calling_output_tool(output: BaseModel) -> FunctionModel:
    """Create a local model that calls the agent's sole structured-output tool."""
    def respond(_messages, agent_info):
        assert len(agent_info.output_tools) == 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=agent_info.output_tools[0].name,
                    args=output.model_dump(),
                )
            ]
        )

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
