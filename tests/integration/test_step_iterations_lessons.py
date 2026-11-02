import asyncio
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_representation import (
    DataRepresentationOutput,
    DataRepresentationStep,
)
from agentomics.runtime.iterations_lessons import ITERATIONS_LESSONS_INSTRUCTIONS
from agentomics.runtime.read_write_utils import load_current_iteration_base_prompt, load_current_iteration_index
from agentomics.utils.exceptions import IterationRunFailed
from tests.helpers import create_failing_model, create_model_calling_output_tool, create_runtime_step

MOCK_ITERATIONS_LESSONS_RESPONSE = "- Representation: use one-hot encoding. All sequences have the same length."


def _iterations_lessons_model(received_requests: list[list[ModelMessage]]) -> FunctionModel:
    def respond(messages, _agent_info):
        received_requests.append(messages)
        return ModelResponse(parts=[TextPart(content=MOCK_ITERATIONS_LESSONS_RESPONSE)])

    return FunctionModel(respond)


def _received_parts(messages: list[ModelMessage]) -> list:
    return [part for message in messages for part in message.parts]


def _assert_iterations_lessons_reach_next_step_prompt(config) -> None:
    next_step_prompt = create_runtime_step(DataRepresentationStep, config, model=Mock()).build_user_prompt()
    assert f"<iterations_lessons>\n{MOCK_ITERATIONS_LESSONS_RESPONSE}\n</iterations_lessons>" in next_step_prompt


# Lessons reach the next step's prompt


def test_iterations_lessons_from_successful_step_reach_next_step_prompt(initialized_run_config, prepared_iteration):
    received_requests: list[list[ModelMessage]] = []
    step = create_runtime_step(
        DataRepresentationStep,
        initialized_run_config,
        model=create_model_calling_output_tool(
            DataRepresentationOutput(representation="One-hot encoded nucleotide sequences.")
        ),
        trace_analysis_model=_iterations_lessons_model(received_requests),
    )

    asyncio.run(step.run())

    assert len(received_requests) == 1
    assert received_requests[0][-1].instructions == ITERATIONS_LESSONS_INSTRUCTIONS.strip()
    iteration = load_current_iteration_index(initialized_run_config)
    assert f"Iteration {iteration}, step {DataRepresentationStep.step_id} just ended." in (
        received_requests[0][-1].parts[-1].content
    )
    step_tool_calls = [part for part in _received_parts(received_requests[0]) if isinstance(part, ToolCallPart)]
    assert any("One-hot encoded nucleotide sequences." in part.args_as_json_str() for part in step_tool_calls)
    _assert_iterations_lessons_reach_next_step_prompt(initialized_run_config)


def test_iterations_lessons_from_failed_step_reach_next_step_prompt(initialized_run_config, prepared_iteration):
    received_requests: list[list[ModelMessage]] = []
    step = create_runtime_step(
        DataRepresentationStep,
        initialized_run_config,
        model=create_failing_model(),
        trace_analysis_model=_iterations_lessons_model(received_requests),
    )

    with pytest.raises(IterationRunFailed):
        asyncio.run(step.run())

    assert len(received_requests) == 1
    assert "runtime error" in received_requests[0][-1].parts[-1].content
    _assert_iterations_lessons_reach_next_step_prompt(initialized_run_config)


# What the lessons agent receives


def test_iterations_lessons_agent_receives_step_prompts_without_adopting_step_system_prompt(
    initialized_run_config, prepared_iteration
):
    received_requests: list[list[ModelMessage]] = []
    step = create_runtime_step(
        DataRepresentationStep,
        initialized_run_config,
        model=create_model_calling_output_tool(
            DataRepresentationOutput(representation="One-hot encoded nucleotide sequences.")
        ),
        trace_analysis_model=_iterations_lessons_model(received_requests),
    )

    asyncio.run(step.run())

    received_parts = _received_parts(received_requests[0])
    user_prompts = [part.content for part in received_parts if isinstance(part, UserPromptPart)]
    assert f"<agent_system_prompt>\n{step.get_system_prompt()}\n</agent_system_prompt>" in user_prompts
    assert load_current_iteration_base_prompt(initialized_run_config) in user_prompts
    assert not any(isinstance(part, SystemPromptPart) for part in received_parts)


# What does not become a lesson


def test_codebase_error_does_not_update_iterations_lessons(initialized_run_config, prepared_iteration):
    received_requests: list[list[ModelMessage]] = []
    step = create_runtime_step(
        DataRepresentationStep,
        initialized_run_config,
        model=Mock(),
        trace_analysis_model=_iterations_lessons_model(received_requests),
    )
    step.build_user_prompt = Mock(side_effect=RuntimeError("codebase bug"))

    with pytest.raises(RuntimeError, match="codebase bug"):
        asyncio.run(step.run())

    assert received_requests == []
