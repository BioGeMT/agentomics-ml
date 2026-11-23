from collections import Counter
from decimal import Decimal

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from agentomics.run_logging.token_tracing import enable_token_tracing, token_usage


@pytest.fixture
def token_tracing():
    token_usage.take()
    enable_token_tracing()
    yield token_usage
    Agent.instrument_all(False)
    token_usage.current_step = None
    token_usage.take()


def _answering_model() -> FunctionModel:
    return FunctionModel(
        lambda _messages, _info: ModelResponse(
            parts=[TextPart(content="done")],
            usage=RequestUsage(input_tokens=100, output_tokens=7),
        )
    )


def test_model_calls_are_counted_under_the_step_they_ran_in(token_tracing):
    agent = Agent(_answering_model())

    token_tracing.current_step = "data_exploration"
    agent.run_sync("Explore the data.")
    agent.run_sync("Summarize it.")
    token_tracing.current_step = None
    agent.run_sync("Not part of a step.")

    usage = token_tracing.take()
    assert usage["data_exploration"] == Counter(input_tokens=200, output_tokens=14, unpriced_requests=2)
    assert usage["outside_steps"] == Counter(input_tokens=100, output_tokens=7, unpriced_requests=1)


def test_cost_of_priced_model_calls_is_counted_under_the_step_they_ran_in(token_tracing):
    agent = Agent(
        FunctionModel(
            lambda _messages, _info: ModelResponse(
                parts=[TextPart(content="done")],
                usage=RequestUsage(input_tokens=100, output_tokens=7, cost=Decimal("0.25")),
            )
        )
    )

    token_tracing.current_step = "model_training"
    agent.run_sync("Train the model.")
    agent.run_sync("Train it again.")

    usage = token_tracing.take()["model_training"]
    assert usage["cost"] == pytest.approx(0.5)
    assert usage["unpriced_requests"] == 0


def test_tokens_spent_before_a_failure_are_counted(token_tracing):
    agent = Agent(
        FunctionModel(
            lambda _messages, _info: ModelResponse(
                parts=[ToolCallPart(tool_name="train", args={})],
                usage=RequestUsage(input_tokens=300, output_tokens=20),
            )
        )
    )

    @agent.tool_plain
    def train() -> str:
        raise RuntimeError("training crashed")

    token_tracing.current_step = "model_training"
    with pytest.raises(RuntimeError):
        agent.run_sync("Train the model.")

    assert token_tracing.take()["model_training"] == Counter(
        input_tokens=300, output_tokens=20, tool_calls=1, unpriced_requests=1
    )


def test_tool_calls_are_counted_under_the_step_they_ran_in(token_tracing):
    def model(messages, _info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_files"), ToolCallPart(tool_name="list_files")])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(model))

    @agent.tool_plain
    def list_files() -> str:
        return "train.csv"

    token_tracing.current_step = "data_exploration"
    agent.run_sync("Look at the files.")

    assert token_tracing.take()["data_exploration"]["tool_calls"] == 2


def test_model_calls_that_failed_without_a_response_do_not_make_the_cost_unknown(token_tracing):
    def failing_model(_messages, _info):
        raise RuntimeError("Provider returned an empty response")

    agent = Agent(FunctionModel(failing_model))

    token_tracing.current_step = "data_split"
    with pytest.raises(RuntimeError):
        agent.run_sync("Split the data.")

    assert token_tracing.take()["data_split"]["unpriced_requests"] == 0
