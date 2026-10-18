import asyncio
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agentomics.agents.steps.data_representation import (
    DataRepresentationOutput,
    DataRepresentationStep,
)
from agentomics.utils.exceptions import IterationRunFailed
from tests.helpers import create_model_calling_output_tool

MOCK_ITERATIONS_LESSONS_RESPONSE = "- Representation: use one-hot encoding. All sequences have the same length."


def _failing_model() -> FunctionModel:
    def respond(_messages, _agent_info):
        raise RuntimeError("provider returned finish_reason='error'")

    return FunctionModel(respond)


def _iterations_lessons_model(received_prompts: list[str]) -> FunctionModel:
    def respond(messages, _agent_info):
        received_prompts.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart(content=MOCK_ITERATIONS_LESSONS_RESPONSE)])

    return FunctionModel(respond)


def _create_step(config, step_model, iterations_lessons_model) -> DataRepresentationStep:
    return DataRepresentationStep(config, step_model, iterations_lessons_model, Mock(), [])


def test_iterations_lessons_from_successful_step_reach_next_step_prompt(initialized_run_config, prepared_iteration):
    received_prompts: list[str] = []
    step = _create_step(
        initialized_run_config,
        create_model_calling_output_tool(
            DataRepresentationOutput(representation="One-hot encoded nucleotide sequences.")
        ),
        _iterations_lessons_model(received_prompts),
    )

    asyncio.run(step.run())

    assert len(received_prompts) == 1
    assert "One-hot encoded nucleotide sequences." in received_prompts[0]
    next_step_prompt = _create_step(initialized_run_config, Mock(), Mock()).build_user_prompt()
    assert f"<iterations_lessons>\n{MOCK_ITERATIONS_LESSONS_RESPONSE}\n</iterations_lessons>" in next_step_prompt


def test_iterations_lessons_from_failed_step_reach_next_step_prompt(initialized_run_config, prepared_iteration):
    received_prompts: list[str] = []
    step = _create_step(initialized_run_config, _failing_model(), _iterations_lessons_model(received_prompts))

    with pytest.raises(IterationRunFailed):
        asyncio.run(step.run())

    assert len(received_prompts) == 1
    assert "provider returned finish_reason='error'" in received_prompts[0]
    next_step_prompt = _create_step(initialized_run_config, Mock(), Mock()).build_user_prompt()
    assert f"<iterations_lessons>\n{MOCK_ITERATIONS_LESSONS_RESPONSE}\n</iterations_lessons>" in next_step_prompt
