import asyncio

import pytest
from pydantic_ai import Agent

from agentomics.agents.agent_utils import (
    extract_prompt_messages,
    extract_step_messages,
    fabricate_final_result_messages,
    fabricate_initial_prompt_messages,
    run_agent,
)
from agentomics.agents.steps.data_representation import DataRepresentationOutput
from agentomics.utils.exceptions import IterationRunFailed
from tests.helpers import create_failing_model, create_model_calling_output_tool, message_part_kinds

STEP_USER_PROMPT = "Describe the data representation."


def _message_history() -> list:
    return [
        *fabricate_initial_prompt_messages(system_prompt="System prompt.", user_prompt="Base prompt."),
        *fabricate_final_result_messages({"summary": "Previous step output."}, model_name="test-model"),
    ]


@pytest.mark.parametrize(
    ("model", "expected_part_kinds"),
    [
        (
            create_model_calling_output_tool(DataRepresentationOutput(representation="One-hot encoded sequences.")),
            [["user-prompt"], ["tool-call"], ["tool-return"]],
        ),
        (create_failing_model(), [["user-prompt"]]),
    ],
    ids=["succeeded", "failed"],
)
@pytest.mark.parametrize("message_history", [_message_history(), None], ids=["with_history", "without_history"])
def test_step_messages_exclude_message_history(model, expected_part_kinds, message_history):
    agent = Agent(model=model, output_type=DataRepresentationOutput)
    try:
        run_messages, _ = asyncio.run(
            run_agent(
                agent=agent,
                user_prompt=STEP_USER_PROMPT,
                max_steps=5,
                message_history=message_history,
                verbose=False,
            )
        )
    except IterationRunFailed as error:
        run_messages = error.context_messages

    step_messages = extract_step_messages(run_messages, message_history)

    assert message_part_kinds(step_messages) == expected_part_kinds
    assert step_messages[0].parts[0].content == STEP_USER_PROMPT


@pytest.mark.parametrize(
    ("message_history", "expected_part_kinds"),
    [
        (_message_history(), [["system-prompt", "user-prompt"]]),
        (None, []),
    ],
    ids=["with_history", "without_history"],
)
def test_prompt_messages_keep_only_system_and_base_prompts(message_history, expected_part_kinds):
    assert message_part_kinds(extract_prompt_messages(message_history)) == expected_part_kinds
