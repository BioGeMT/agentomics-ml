import traceback
import datetime
import string
import random

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
import weave

from utils.exceptions import IterationRunFailed
from utils.printing_utils import pretty_print_node


@weave.op(call_display_name=lambda call: f"Agent Step - {call.inputs['output_type'].__name__ if call.inputs.get('output_type', None) else call.inputs['agent']._output_type.__name__}")
async def run_agent(agent: Agent, user_prompt: str, max_steps: int, message_history: list | None, output_type = None, verbose: bool = True, deps=None):
    with capture_run_messages() as messages:
        try:
            async with agent.iter(
                user_prompt=user_prompt,
                usage_limits=UsageLimits(request_limit=max_steps),
                output_type=output_type,
                message_history=message_history,
                deps=deps,
            ) as agent_run:
                async for node in agent_run:
                    if(verbose):
                        pretty_print_node(node)
                return agent_run.result.all_messages(), agent_run.result.output

        except Exception as e:
            trace = traceback.format_exc()
            if(verbose):
                print('--------------- ERROR TRACEBACK ---------------')
                print('Agent run failed', trace)
                print('--------------- ERROR TRACEBACK ---------------')
            raise IterationRunFailed(
                message="Run didnt finish properly", 
                context_messages=messages,
                exception_trace=trace,
            )

def fabricate_final_result_messages(structured_output, model_name):
    if hasattr(structured_output, "model_dump"):
        output_dict = structured_output.model_dump()
    elif isinstance(structured_output, dict):
        output_dict = structured_output
    else:
        output_dict = vars(structured_output)
    tool_call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
    response_msg = ModelResponse(
        parts=[
            TextPart(content='', part_kind='text'),
            ToolCallPart(tool_name="final_result", args = output_dict, tool_call_id=tool_call_id, part_kind='tool-call'),
        ],
        timestamp=datetime.datetime.now(),
        kind='response', 
        model_name=model_name,
    )
    request_msg = ModelRequest(
        parts=[
            ToolReturnPart(tool_name="final_result", content="Final result processed.", tool_call_id=tool_call_id, 
                           part_kind='tool-return', timestamp=datetime.datetime.now())
        ],
        kind='request',
    )
    return [response_msg, request_msg]

def fabricate_initial_prompt_messages(system_prompt: str, user_prompt: str) -> list[ModelMessage]:
    return [
        ModelRequest(
            parts=[
                SystemPromptPart(content=system_prompt),
                UserPromptPart(content=user_prompt),
            ],
            kind='request',
        )
    ]
