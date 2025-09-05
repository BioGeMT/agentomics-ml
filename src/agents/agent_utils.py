import traceback
from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.usage import UsageLimits
from utils.exceptions import IterationRunFailed
from utils.printing_utils import pretty_print_node

async def run_agent(agent: Agent, user_prompt: str, max_steps: int, message_history: list | None, output_type: BaseModel = None, verbose: bool = True):
    with capture_run_messages() as messages:
        try:
            async with agent.iter(
                user_prompt=user_prompt,
                usage_limits=UsageLimits(request_limit=max_steps),
                output_type=output_type,
                message_history=message_history,
            ) as agent_run:
                async for node in agent_run:
                    if(verbose):
                        pretty_print_node(node)
                return agent_run.result.all_messages(), agent_run.result.data

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