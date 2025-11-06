import traceback
from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.usage import UsageLimits
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