import asyncio
import shutil
import subprocess
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_representation import DataRepresentationStep
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.iteration_plan import IterationPlanOutput, IterationPlanStep
from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.runtime.read_write_utils import (
    load_current_iteration_index,
    archive_current_iteration,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.step_outputs import load_step_outputs
from agentomics.utils.config import Config
from tests.helpers import create_runtime_step

NOTES_MARKER = "NOTES-MARKER-42"


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
        self.tool_names: list[str] = []
        self.tool_returns: list[str] = []

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
            self.tool_names = [tool.name for tool in agent_info.function_tools]
            self.tool_returns = [
                str(part.content)
                for message in messages
                for part in message.parts
                if part.part_kind == "tool-return"
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
    step = IterationPlanStep(config, Mock(), planner.model(), provider, [])
    asyncio.run(step.run())


def _finish_iteration(config, iteration: int) -> None:
    # Marked failed so the planner prompt needs no history of other steps.
    update_current_iteration_state(config, ended_at=110.0, status="failed")
    archive_current_iteration(config, iteration)
    prepare_iteration_workspace(config, iteration=iteration + 1)


@pytest.fixture
def supplementary_folder(initialized_run_config):
    folder = initialized_run_config.dataset_dir / SUPPLEMENTARY_DIR_NAME
    folder.mkdir(exist_ok=True)
    (folder / "README.md").write_text("index", encoding="utf-8")


def _notes_text(config, iteration: int) -> str:
    matches = list(config.iteration_dir(iteration).glob("iteration_plan/*notes*"))
    assert len(matches) == 1
    return matches[0].read_text(encoding="utf-8")


def test_notes_field_is_in_planner_schema_but_not_in_serialised_output():
    assert "notes" in IterationPlanOutput.model_json_schema()["properties"]
    output = IterationPlanOutput(**_plan_args(notes=NOTES_MARKER))
    assert "notes" not in output.model_dump()


def test_planner_notes_are_saved_separately_and_injected_into_next_prompt(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    first = ScriptedPlanner(_plan_args(notes=NOTES_MARKER))
    _run_planner(initialized_run_config, first)
    _finish_iteration(initialized_run_config, 0)

    assert NOTES_MARKER not in first.prompts[0]
    assert _notes_text(initialized_run_config, 0) == NOTES_MARKER
    saved_output = initialized_run_config.iteration_dir(0) / "iteration_plan" / Config.STEP_OUTPUT_FILENAME
    assert NOTES_MARKER not in saved_output.read_text(encoding="utf-8")

    second = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, second)
    assert NOTES_MARKER in second.prompts[0]


def test_absent_notes_are_kept_and_present_notes_replace_in_full(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="old notes")))
    _finish_iteration(initialized_run_config, 0)

    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args()))
    _finish_iteration(initialized_run_config, 1)
    assert _notes_text(initialized_run_config, 1) == "old notes"

    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="new notes")))
    _finish_iteration(initialized_run_config, 2)

    third = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, third)
    assert "new notes" in third.prompts[0]
    assert "old notes" not in third.prompts[0]


def test_over_cap_notes_are_sent_back_to_the_planner_for_retry(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    initialized_run_config.planner_notes_max_chars = 10
    planner = ScriptedPlanner(_plan_args(notes="x" * 11), _plan_args(notes="x" * 10))

    _run_planner(initialized_run_config, planner)

    assert len(planner.prompts) == 2
    notes_files = list(initialized_run_config.current_iteration_dir.glob("iteration_plan/*notes*"))
    assert [f.read_text(encoding="utf-8") for f in notes_files] == ["x" * 10]


def test_notes_never_reach_iteration_agent_history(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))

    step = create_runtime_step(DataRepresentationStep, initialized_run_config, model=Mock())
    history_text = str(step.get_message_history())
    assert "one-hot" in history_text
    assert NOTES_MARKER not in history_text
    assert all("notes" not in output.model_dump() for output in load_step_outputs(initialized_run_config))


def test_notes_are_committed_in_the_planner_step_checkpoint(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=initialized_run_config.workspace_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "iteration_plan/planner_notes" in tracked


def test_planner_runs_without_notes_file_or_supplementary_folder(
    initialized_run_config, prepared_iteration
):
    planner = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, planner)
    assert len(planner.prompts) == 1


def test_planner_gets_the_supplementary_tool_when_the_folder_exists(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    planner = ScriptedPlanner(
        ("supplementary", {"operation": "read", "path": "README.md"}),
        _plan_args(),
    )
    _run_planner(initialized_run_config, planner)

    assert planner.tool_names == ["supplementary"]
    assert "index" in planner.tool_returns[0]


def test_planner_has_no_tool_without_supplementary_folder(initialized_run_config, prepared_iteration):
    shutil.rmtree(initialized_run_config.dataset_dir / SUPPLEMENTARY_DIR_NAME)
    planner = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, planner)
    assert planner.tool_names == []


def test_exhausted_tool_budget_is_not_cut_off_by_the_request_limit(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    config = initialized_run_config
    config.supplementary_tool_max_calls = 3
    config.max_steps = 2  # below the tool budget: the planner must not be capped by it
    call = ("supplementary", {"operation": "list"})
    planner = ScriptedPlanner(call, call, call, call, _plan_args(notes=NOTES_MARKER))

    _run_planner(config, planner)

    assert "budget is spent" in planner.tool_returns[-1]
    notes_files = list(config.current_iteration_dir.glob("iteration_plan/*notes*"))
    assert [f.read_text(encoding="utf-8") for f in notes_files] == [NOTES_MARKER]
