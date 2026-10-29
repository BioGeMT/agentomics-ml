import asyncio
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_representation import DataRepresentationStep
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.iteration_plan import (
    IterationPlanStep,
)
from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.runtime.git_checkpoints import commit_iteration_end
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    load_config_from_run_dir,
    load_current_iteration_index,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.setup_fork import fork_run
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


def _load_notes(config):
    return IterationPlanStep(config, Mock(), Mock(), Mock(), [])._load_notes()


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
    commit_iteration_end(config, iteration)
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


def test_notes_never_reach_iteration_agent_history(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))

    step = create_runtime_step(DataRepresentationStep, initialized_run_config, model=Mock())
    history_text = str(step.get_message_history())
    assert "one-hot" in history_text
    assert NOTES_MARKER not in history_text
    assert all("notes" not in output.model_dump() for output in load_step_outputs(initialized_run_config))


def test_empty_string_notes_replace_with_empty_while_absent_keeps(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="old notes")))
    _finish_iteration(initialized_run_config, 0)

    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="")))
    _finish_iteration(initialized_run_config, 1)
    assert _notes_text(initialized_run_config, 1) == ""

    third = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, third)
    assert "old notes" not in third.prompts[0]


# --- fork inheritance -------------------------------------------------------------------------


@pytest.fixture
def three_iterations_of_notes(initialized_run_config, prepared_iteration, supplementary_folder):
    config = initialized_run_config
    for iteration, marker in enumerate(("NOTES-A", "NOTES-B", "NOTES-C")):
        _run_planner(config, ScriptedPlanner(_plan_args(notes=marker)))
        _finish_iteration(config, iteration)
    # Leave the source as a finished run, without the uncommitted workspace of a next iteration.
    shutil.rmtree(config.current_iteration_dir)
    return config


def _fork(source_config, tmp_path, monkeypatch, *, step, iteration):
    # Environment rebuilding needs conda and is irrelevant to the notes.
    monkeypatch.setattr("agentomics.runtime.setup_fork.ensure_environment_from_descriptor", lambda *a, **k: None)
    target = tmp_path / "forked-workspace"
    fork_run(
        source_workspace_dir=Path(source_config.workspace_dir),
        target_agent_id="forked",
        target_workspace_dir=target,
        fork_from_step=step,
        fork_from_iteration=iteration,
    )
    return load_config_from_run_dir(target / Config.RUN_DIRNAME)


def test_fork_from_earlier_iteration_sees_notes_as_of_that_checkpoint(
    three_iterations_of_notes, tmp_path, monkeypatch
):
    fork_config = _fork(three_iterations_of_notes, tmp_path, monkeypatch, step="end", iteration=0)

    assert _load_notes(fork_config) == "NOTES-A"
    assert not fork_config.iteration_dir(1).exists()

    prepare_iteration_workspace(fork_config, iteration=1)
    # The notes block of the next planner prompt (the full prompt needs split state a fork does not carry in git).
    prompt = IterationPlanStep(fork_config, Mock(), Mock(), Mock(), [])._build_notes_info()
    assert "NOTES-A" in prompt
    assert "NOTES-B" not in prompt and "NOTES-C" not in prompt
