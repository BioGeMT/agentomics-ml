import asyncio
import json
from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.agent_utils import run_agent
from agentomics.runtime import read_write_utils, ui_events


def _scripted_agent() -> Agent:
    def respond(messages, info):
        if sum(isinstance(message, ModelResponse) for message in messages) == 0:
            return ModelResponse(parts=[ToolCallPart("bash", {"command": "ls"})])
        return ModelResponse(parts=[TextPart("Found the data.")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def bash(command: str) -> str:
        """Run a command."""
        return "data.csv"

    return agent


def _event_types(events_path: Path) -> list[str]:
    return [json.loads(line)["type"] for line in events_path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def first_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(read_write_utils, "load_current_iteration_index", lambda config: 0)


@pytest.fixture
def events_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ui_events.EVENTS_FILENAME
    path.write_text("")
    monkeypatch.setattr(ui_events, "_events_path", path)
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", True)
    return path


def test_events_are_not_recorded_when_ui_is_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ui_events, "_events_path", None)
    # Disabled, a step reads nothing: not even the iteration index for its name.
    monkeypatch.setattr(read_write_utils, "load_current_iteration_index", lambda config: pytest.fail("read while disabled"))

    assert isinstance(ui_events.agent_events(), ui_events._DisabledAgentEvents)
    with ui_events.step(None, "Planning"):
        pass


def test_agent_run_is_recorded_as_ag_ui_events(events_path: Path):
    with ui_events.step(None, "Data Exploration"):
        asyncio.run(run_agent(_scripted_agent(), "go", max_steps=5, message_history=None, verbose=False))

    assert _event_types(events_path) == [
        "STEP_STARTED",
        # AG-UI announces the assistant message the tool call belongs to, even without text.
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "CUSTOM",  # usage of the model response that made the tool call
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "CUSTOM",
        "STEP_FINISHED",
    ]


def test_each_model_response_reports_its_usage(events_path: Path):
    with ui_events.step(None, "Data Exploration"):
        asyncio.run(run_agent(_scripted_agent(), "go", max_steps=5, message_history=None, verbose=False))

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    usage = [
        ui_events.Usage.model_validate(event["value"])
        for event in events
        if event["type"] == "CUSTOM" and event["name"] == ui_events.USAGE_EVENT
    ]
    assert len(usage) == 2
    assert all(u.input_tokens > 0 and u.output_tokens > 0 for u in usage)
    # The test model has no price, so the cost is unknown rather than zero.
    assert all(u.cost_usd is None for u in usage)


def test_failed_step_is_still_finished(events_path: Path):
    with pytest.raises(RuntimeError):
        with ui_events.step(None, "Training"):
            raise RuntimeError("training crashed")

    assert _event_types(events_path) == ["STEP_STARTED", "STEP_FINISHED"]


def test_run_stream_is_a_no_op_without_the_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ui_events.UI_EVENTS_ENV, raising=False)
    monkeypatch.setattr(ui_events, "_events_path", None)

    with ui_events.run_stream(tmp_path, "run-1") as result:
        result["exit_code"] = 0

    assert not (tmp_path / "logs" / ui_events.EVENTS_FILENAME).exists()


def test_run_stream_brackets_the_run_and_records_its_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ui_events.UI_EVENTS_ENV, "1")
    monkeypatch.setattr(ui_events, "_events_path", None)

    with ui_events.run_stream(tmp_path, "run-1") as result:
        with ui_events.step(None, "Planning"):
            pass
        result["exit_code"] = 0

    events = [json.loads(line) for line in (tmp_path / "logs" / ui_events.EVENTS_FILENAME).read_text().splitlines()]
    assert [event["type"] for event in events] == ["RUN_STARTED", "STEP_STARTED", "STEP_FINISHED", "RUN_FINISHED"]
    assert events[-1]["result"] == {"exit_code": 0}
