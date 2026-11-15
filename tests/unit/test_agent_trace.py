import asyncio
from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.agent_utils import run_agent
from agentomics.utils.exceptions import AgentRunFailed


@pytest.fixture(autouse=True)
def allow_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", True)


def _scripted_agent(fail_after_tool: bool = False) -> Agent:
    def respond(messages, info):
        if sum(isinstance(message, ModelResponse) for message in messages) == 0:
            return ModelResponse(parts=[ToolCallPart("bash", {"command": "ls"})])
        if fail_after_tool:
            raise RuntimeError("model unavailable")
        return ModelResponse(parts=[TextPart("Found the data.")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def bash(command: str) -> str:
        """Run a command."""
        return "data.csv"

    return agent


def _trace(path: Path) -> list[ModelMessage]:
    adapter = TypeAdapter(ModelMessage)
    return [adapter.validate_json(line) for line in path.read_text().splitlines()]


def _part_kinds(messages: list[ModelMessage]) -> list[list[str]]:
    return [[part.part_kind for part in message.parts] for message in messages]


def test_trace_records_the_runs_messages_as_they_happen(tmp_path: Path):
    trace_path = tmp_path / "traces" / "iteration_0" / "data_exploration.jsonl"
    earlier = [ModelRequest(parts=[UserPromptPart("an earlier step")])]

    asyncio.run(run_agent(_scripted_agent(), "go", max_steps=5, message_history=earlier, verbose=False, trace_path=trace_path))

    messages = _trace(trace_path)
    # The new prompt, the tool call, its result, the answer: not the earlier steps' history.
    assert _part_kinds(messages) == [["user-prompt"], ["tool-call"], ["tool-return"], ["text"]]
    assert messages[0].parts[0].content == "go"
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    assert all(response.usage.input_tokens > 0 for response in responses)
    # The test model has no price, so the cost stays unknown rather than zero.
    assert all(response.usage.cost is None for response in responses)


def test_trace_keeps_what_happened_before_a_failure(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"

    with pytest.raises(AgentRunFailed):
        asyncio.run(run_agent(_scripted_agent(fail_after_tool=True), "go", max_steps=5, message_history=None, verbose=False, trace_path=trace_path))

    assert _part_kinds(_trace(trace_path)) == [["user-prompt"], ["tool-call"], ["tool-return"]]


def test_a_new_run_starts_a_fresh_trace(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("left over from an earlier attempt\n")

    asyncio.run(run_agent(_scripted_agent(), "go", max_steps=5, message_history=None, verbose=False, trace_path=trace_path))

    assert _part_kinds(_trace(trace_path))[0] == ["user-prompt"]


def test_no_trace_is_written_without_a_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    asyncio.run(run_agent(_scripted_agent(), "go", max_steps=5, message_history=None, verbose=False))

    assert list(tmp_path.iterdir()) == []


def test_a_priced_models_cost_is_filled_in():
    from decimal import Decimal

    from pydantic_ai.usage import RequestUsage

    from agentomics.agents.agent_utils import _with_cost

    response = ModelResponse(
        parts=[TextPart("hi")],
        usage=RequestUsage(input_tokens=1000, output_tokens=500),
        model_name="gpt-4o",
        provider_name="openai",
    )

    assert _with_cost(response).usage.cost == Decimal(str(response.cost().total_price))
    assert _with_cost(response).usage.cost > 0
