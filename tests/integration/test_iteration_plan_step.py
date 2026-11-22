import asyncio
import shutil
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.iteration_plan import (
    SUPPLEMENTARY_MAX_TOOL_CALLS,
    IterationPlanStep,
)
from agentomics.runtime.read_write_utils import load_current_iteration_index
from tests.helpers import create_runtime_step


def _plan_args(**overrides) -> dict:
    args = {
        "data_exploration_instructions": "skip",
        "data_split_instructions": "skip",
        "data_representation_instructions": "one-hot",
        "model_architecture_instructions": "small cnn",
        "model_training_instructions": "adam",
        "model_inference_instructions": "predict",
        "prediction_exploration_instructions": "skip",
        "other_instructions": None,
    }
    return args | overrides


class ScriptedPlanner:
    """Model that replies with queued planner outputs and records prompts."""

    def __init__(self, *replies: dict | tuple) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.last_request_tool_names: list[str] = []
        self.tool_results: list[str] = []

    def model(self) -> FunctionModel:
        def respond(messages, agent_info):
            self.prompts.append(
                "\n".join(
                    str(part.content)
                    for message in messages
                    for part in message.parts
                    if part.part_kind == "user-prompt"
                )
            )
            self.last_request_tool_names = [tool.name for tool in agent_info.function_tools]
            # Tool returns and tool errors sent back for a retry, in order.
            self.tool_results = [
                str(part.content)
                for message in messages
                for part in message.parts
                if part.part_kind in ("tool-return", "retry-prompt")
            ]
            reply = self.replies.pop(0)
            if isinstance(reply, tuple):  # (tool_name, args) is a call to the planner's tool
                return ModelResponse(parts=[ToolCallPart(tool_name=reply[0], args=reply[1])])
            return ModelResponse(parts=[ToolCallPart(tool_name=agent_info.output_tools[0].name, args=reply)])

        return FunctionModel(respond)


def _run_planner(config, planner: ScriptedPlanner) -> None:
    iteration = load_current_iteration_index(config)
    create_runtime_step(DataSplitStep, config, model=Mock()).on_iteration_start(iteration)
    provider = Mock()
    provider.get_high_reasoning_model_settings.return_value = {}
    step = create_runtime_step(IterationPlanStep, config, model=Mock(), iteration_plan_model=planner.model(), provider=provider)
    asyncio.run(step.run())


@pytest.fixture
def supplementary_readme(initialized_run_config):
    # The standard dataset already has a supplementary folder.
    (initialized_run_config.supplementary_dir / "README.md").write_text("index", encoding="utf-8")


# --- reading the supplementary folder ---------------------------------------------------------


def test_planner_lists_and_reads_supplementary_folder(
    initialized_run_config, prepared_iteration, supplementary_readme
):
    planner = ScriptedPlanner(("list_directory", {"path": "."}), ("read_file", {"path": "README.md"}), _plan_args())
    _run_planner(initialized_run_config, planner)

    assert "README.md" in planner.tool_results[0]
    assert "index" in planner.tool_results[1]


def test_planner_reads_supplementary_folder_that_is_a_symlink(
    initialized_run_config, prepared_iteration, supplementary_readme, tmp_path
):
    # Dataset preparation links the prepared dataset's supplementary folder to the source dataset's.
    folder = initialized_run_config.supplementary_dir
    source = tmp_path / "source_supplementary"
    shutil.move(folder, source)
    folder.symlink_to(source, target_is_directory=True)
    planner = ScriptedPlanner(("read_file", {"path": "README.md"}), _plan_args())
    _run_planner(initialized_run_config, planner)

    assert "index" in planner.tool_results[0]


def test_planner_cannot_read_outside_supplementary_folder(
    initialized_run_config, prepared_iteration, supplementary_readme
):
    (initialized_run_config.dataset_dir / "secret.txt").write_text("SECRET-CONTENT", encoding="utf-8")
    planner = ScriptedPlanner(("read_file", {"path": "../secret.txt"}), _plan_args())
    _run_planner(initialized_run_config, planner)

    assert planner.replies == []
    assert "SECRET-CONTENT" not in planner.tool_results[0]


def test_without_supplementary_folder_planner_has_no_tools(initialized_run_config, prepared_iteration):
    shutil.rmtree(initialized_run_config.supplementary_dir)
    planner = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, planner)

    assert planner.last_request_tool_names == []


def test_planner_reads_a_pdf_as_a_file(initialized_run_config, prepared_iteration, supplementary_readme):
    (initialized_run_config.supplementary_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n%\xe2\xe3\n")
    planner = ScriptedPlanner(("read_file", {"path": "paper.pdf"}), _plan_args())
    _run_planner(initialized_run_config, planner)

    assert "%PDF-1.4" in planner.tool_results[0]


def test_planner_reading_to_its_call_limit_loses_its_tools_and_gives_a_plan_despite_a_low_step_limit(
    initialized_run_config, prepared_iteration, supplementary_readme
):
    initialized_run_config.max_steps = 3  # The planner's step limit must not depend on the iteration agents' max_steps.
    reads = [("read_file", {"path": "README.md"})] * SUPPLEMENTARY_MAX_TOOL_CALLS
    planner = ScriptedPlanner(*reads, _plan_args())
    _run_planner(initialized_run_config, planner)

    assert planner.replies == []
    assert planner.last_request_tool_names == []  # The request giving the plan was offered no file tools.


def test_planner_loses_its_tools_once_a_request_reaches_the_token_limit(
    initialized_run_config, prepared_iteration, supplementary_readme, monkeypatch
):
    # Any model request uses at least one input token.
    monkeypatch.setattr("agentomics.agents.steps.iteration_plan.SUPPLEMENTARY_MAX_CONTEXT_TOKENS", 1)
    planner = ScriptedPlanner(("read_file", {"path": "README.md"}), _plan_args())
    _run_planner(initialized_run_config, planner)

    assert planner.replies == []
    assert planner.last_request_tool_names == []  # The request giving the plan was offered no file tools.
