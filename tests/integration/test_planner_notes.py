import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_representation import DataRepresentationStep
from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.iteration_plan import IterationPlanOutput, IterationPlanStep
from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.runtime.git_checkpoints import commit_iteration_end
from agentomics.runtime.planner_notes import load_current_planner_notes
from agentomics.runtime.read_write_utils import (
    load_config_from_run_dir,
    load_current_iteration_index,
    archive_current_iteration,
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


def test_notes_field_is_in_planner_schema_but_not_in_serialised_output():
    notes_schema = IterationPlanOutput.model_json_schema()["properties"]["notes"]
    assert "maxLength" in json.dumps(notes_schema) and "6000" in json.dumps(notes_schema)
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
    planner = ScriptedPlanner(
        _plan_args(notes="x" * (6001)), _plan_args(notes="x" * 6000)
    )

    _run_planner(initialized_run_config, planner)

    assert len(planner.prompts) == 2
    notes_files = list(initialized_run_config.current_iteration_dir.glob("iteration_plan/*notes*"))
    assert [f.read_text(encoding="utf-8") for f in notes_files] == ["x" * 6000]


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
        ("supplementary_read", {"path": "README.md"}),
        _plan_args(),
    )
    _run_planner(initialized_run_config, planner)

    assert planner.tool_names == ["supplementary_list", "supplementary_search", "supplementary_read"]
    assert "index" in planner.tool_returns[0]


def test_planner_has_no_tool_without_supplementary_folder(initialized_run_config, prepared_iteration):
    shutil.rmtree(initialized_run_config.dataset_dir / SUPPLEMENTARY_DIR_NAME)
    planner = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, planner)
    assert planner.tool_names == []


def test_exhausted_tool_budget_ends_reading_and_planner_still_saves_notes(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    config = initialized_run_config
    config.supplementary_tool_max_calls = 3
    call = ("supplementary_list", {})
    planner = ScriptedPlanner(call, call, call, call, _plan_args(notes=NOTES_MARKER))

    _run_planner(config, planner)

    assert "budget is spent" in planner.tool_returns[-1]
    notes_files = list(config.current_iteration_dir.glob("iteration_plan/*notes*"))
    assert [f.read_text(encoding="utf-8") for f in notes_files] == [NOTES_MARKER]


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


def test_notes_are_neither_saved_nor_injected_without_supplementary_folder(
    initialized_run_config, prepared_iteration
):
    config = initialized_run_config
    shutil.rmtree(config.dataset_dir / SUPPLEMENTARY_DIR_NAME)
    planner = ScriptedPlanner(_plan_args(notes=NOTES_MARKER))
    _run_planner(config, planner)

    assert list(config.current_iteration_dir.glob("iteration_plan/*notes*")) == []
    assert "your_notes" not in planner.prompts[0]


def test_failed_iteration_keeps_its_notes_and_unfinished_planner_is_skipped(
    initialized_run_config, prepared_iteration, supplementary_folder
):
    config = initialized_run_config
    _run_planner(config, ScriptedPlanner(_plan_args(notes="from failed iteration")))
    _finish_iteration(config, 0)  # archived with status failed
    # Iteration 1: planner step never completed, so it has no notes file.
    update_current_iteration_state(config, ended_at=120.0, status="failed")
    archive_current_iteration(config, 1)
    prepare_iteration_workspace(config, iteration=2)

    assert load_current_planner_notes(config) == "from failed iteration"


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

    assert load_current_planner_notes(fork_config) == "NOTES-A"
    assert not fork_config.iteration_dir(1).exists()

    prepare_iteration_workspace(fork_config, iteration=1)
    # The notes block of the next planner prompt (the full prompt needs split state a fork does not carry in git).
    prompt = IterationPlanStep(fork_config, Mock(), Mock(), Mock(), [])._build_notes_info()
    assert "NOTES-A" in prompt
    assert "NOTES-B" not in prompt and "NOTES-C" not in prompt


def test_fork_from_latest_checkpoint_inherits_latest_notes(three_iterations_of_notes, tmp_path, monkeypatch):
    fork_config = _fork(three_iterations_of_notes, tmp_path, monkeypatch, step="end", iteration=2)

    assert load_current_planner_notes(fork_config) == "NOTES-C"


def test_fork_from_the_planner_step_keeps_that_iterations_notes(
    three_iterations_of_notes, tmp_path, monkeypatch
):
    fork_config = _fork(three_iterations_of_notes, tmp_path, monkeypatch, step="iteration_plan", iteration=1)

    # The iteration is still in progress at that checkpoint; its notes live in the current iteration.
    assert (fork_config.current_iteration_dir / "iteration_plan" / "planner_notes.md").read_text() == "NOTES-B"
    assert not fork_config.iteration_dir(2).exists()
    # Once the forked iteration is archived, its notes are the current notes.
    update_current_iteration_state(fork_config, ended_at=200.0, status="failed")
    archive_current_iteration(fork_config, 1)
    assert load_current_planner_notes(fork_config) == "NOTES-B"
