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
    PLANNER_NOTES_MAX_CHARS,
    SUPPLEMENTARY_MAX_TOOL_CALLS,
    IterationPlanStep,
)
from agentomics.runtime.git_checkpoints import commit_iteration_end
from agentomics.runtime.read_write_utils import (
    archive_current_iteration,
    load_config_from_run_dir,
    load_current_iteration_index,
    update_current_iteration_state,
)
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.runtime.setup_fork import fork_run
from agentomics.utils.config import Config
from agentomics.utils.exceptions import StepFailed
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

    def __init__(self, *replies: dict | tuple | Exception) -> None:
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
            if isinstance(reply, Exception):  # The model request fails.
                raise reply
            if isinstance(reply, tuple):  # (tool_name, args) is a call to the planner's tool
                return ModelResponse(parts=[ToolCallPart(tool_name=reply[0], args=reply[1])])
            # Plan replies are dicts, not IterationPlanOutput: its model_dump() would drop the excluded notes field.
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


# --- planner notes ----------------------------------------------------------------------------


def _finish_iteration(config, iteration: int) -> None:
    # Mirrors the end of run_lifecycle._run_iteration, which is private; run_agentomics would also build a conda environment.
    # Marked failed so the planner prompt needs no history of other steps.
    update_current_iteration_state(config, ended_at=110.0, status="failed")
    archive_current_iteration(config, iteration)
    commit_iteration_end(config, iteration)
    prepare_iteration_workspace(config, iteration=iteration + 1)


def _notes_text(config, iteration: int) -> str:
    return (config.iteration_dir(iteration) / IterationPlanStep.step_id / Config.PLANNER_NOTES_FILENAME).read_text(encoding="utf-8")


def test_notes_never_reach_iteration_agent_history(initialized_run_config, prepared_iteration):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))

    step = create_runtime_step(DataRepresentationStep, initialized_run_config, model=Mock())
    history_text = str(step.get_initial_message_history())
    assert "one-hot" in history_text
    assert NOTES_MARKER not in history_text
    saved_plan = initialized_run_config.archived_step_dir(IterationPlanStep.step_id) / Config.STEP_OUTPUT_FILENAME
    assert NOTES_MARKER not in saved_plan.read_text(encoding="utf-8")


def test_empty_string_notes_clear_previous_notes(initialized_run_config, prepared_iteration):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="old notes")))
    _finish_iteration(initialized_run_config, 0)

    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="")))
    _finish_iteration(initialized_run_config, 1)
    assert _notes_text(initialized_run_config, 1) == ""

    third = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, third)
    assert "old notes" not in third.prompts[0]


def test_absent_notes_keep_previous_notes(initialized_run_config, prepared_iteration):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes="kept notes")))
    _finish_iteration(initialized_run_config, 0)

    second = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, second)
    _finish_iteration(initialized_run_config, 1)

    assert "kept notes" in second.prompts[0]
    assert _notes_text(initialized_run_config, 1) == "kept notes"


def test_notes_survive_an_iteration_whose_planner_failed(initialized_run_config, prepared_iteration):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))
    _finish_iteration(initialized_run_config, 0)
    with pytest.raises(StepFailed):
        _run_planner(initialized_run_config, ScriptedPlanner(RuntimeError("model unavailable")))
    _finish_iteration(initialized_run_config, 1)

    third = ScriptedPlanner(_plan_args())
    _run_planner(initialized_run_config, third)

    assert NOTES_MARKER in third.prompts[0]


def test_overlong_notes_are_sent_back_to_the_planner(initialized_run_config, prepared_iteration):
    planner = ScriptedPlanner(
        _plan_args(notes="x" * (PLANNER_NOTES_MAX_CHARS + 1)),
        _plan_args(notes="short notes"),
    )
    _run_planner(initialized_run_config, planner)
    _finish_iteration(initialized_run_config, 0)

    assert planner.replies == []
    assert _notes_text(initialized_run_config, 0) == "short notes"


def test_planner_keeps_notes_without_a_supplementary_folder(initialized_run_config, prepared_iteration):
    shutil.rmtree(initialized_run_config.supplementary_dir)
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))
    _finish_iteration(initialized_run_config, 0)

    assert _notes_text(initialized_run_config, 0) == NOTES_MARKER


@pytest.fixture
def three_iterations_of_notes(initialized_run_config, prepared_iteration):
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

    assert _notes_text(fork_config, 0) == "NOTES-A"
    assert not fork_config.iteration_dir(1).exists()

    prepare_iteration_workspace(fork_config, iteration=1)
    # The planner prompt needs the data splits. No split step ran in the source run, so its splits folder is empty
    # and git, which does not track empty folders, does not carry it into the fork.
    shutil.copytree(three_iterations_of_notes.shared_dir / "splits", fork_config.shared_dir / "splits")
    planner = ScriptedPlanner(_plan_args())
    _run_planner(fork_config, planner)
    assert "NOTES-A" in planner.prompts[0]
    assert "NOTES-B" not in planner.prompts[0] and "NOTES-C" not in planner.prompts[0]


def test_fork_from_the_planner_step_keeps_the_notes_of_that_step(
    initialized_run_config, prepared_iteration, tmp_path, monkeypatch
):
    _run_planner(initialized_run_config, ScriptedPlanner(_plan_args(notes=NOTES_MARKER)))

    fork_config = _fork(initialized_run_config, tmp_path, monkeypatch, step=IterationPlanStep.step_id, iteration=0)

    notes_path = fork_config.archived_step_dir(IterationPlanStep.step_id) / Config.PLANNER_NOTES_FILENAME
    assert notes_path.read_text(encoding="utf-8") == NOTES_MARKER
