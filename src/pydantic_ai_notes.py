import asyncio
from dataclasses import dataclass
from typing import Union
from litellm import provider_list
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, Tool, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.providers.google_gla import GoogleGLAProvider
import dotenv
import os
from pydantic_ai.models.gemini import GeminiModel
from pydantic_graph import BaseNode, GraphRunContext
from sklearn.metrics import adjusted_rand_score
from run_logging.wandb import setup_logging
from rich.console import Console
from rich.live import Live #TODO needed?
from rich.markdown import Markdown

# import debugpy
# This can be done as console arg as well, needs launch.json config
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()

dotenv.load_dotenv()
# setup_logging(config={"tags":["pydantic_testing"],"agent_id":"pyai_test_0"}, api_key=os.environ["WANDB_API_KEY"])

@dataclass
class Deps:
    name: str

class MagicResult(BaseModel):
    type: int #result should be types (used as hint for agent)

class TextResult(BaseModel):
    response: str

class WeatherResult(BaseModel):
    rains: bool

# The agent gets both, the docstring and the arg description
def do_nothing(ctx, password: str): #If using a parameter, should have a type (used as hint for agent)
    """
    This function returns the true big magic number.

    Args:
        ctx: The context in which the function is called.
        password: The password to access the magic number.
    """
    raise ModelRetry("Retry please")
    return 11

def get_magic_number(ctx, password: str): #If using a parameter, should have a type (used as hint for agent)
    """
    This function returns the magic number.

    Args:
        ctx: The context in which the function is called.
        password: The password to access the magic number.
    """
    return 12

magic_tool = Tool(function=do_nothing, takes_ctx=True, max_retries=1, description=None, require_parameter_descriptions=True)
magic_tool2 = Tool(function=get_magic_number, takes_ctx=True, max_retries=1, description=None, require_parameter_descriptions=True)


model = GeminiModel(
    model_name="gemini-2.0-flash-lite",
    provider=GoogleGLAProvider(api_key=os.getenv("VLASTA_GEMINI_API_KEY")), 
    
)
agent = Agent(  
    model=model,
    system_prompt='Be concise, reply with one sentence. If you dont know the answer, ask the user for clatification.', 
    deps_type=Deps,
    result_type=Union[WeatherResult, str], #Gives the model the option to return both 
    tools=[magic_tool, magic_tool2],
    # model_settings={'temperature':0.0} #Can be passed here for global settings
)
# print(agent._function_tools) #Way to check the function jsons for the agent
# @agent.tool
# def get_magic_number(ctx): #If using a parameter, should have a type (used as hint for agent)
#     raise ModelRetry("Retry please")
#     return 11


# Can be more complex, prompt model to retry, etc https://ai.pydantic.dev/results/#result-validators-functions
# @agent.result_validator
# def validate_result(ctx: RunContext[Deps], result: MagicResult):
#     # print(ctx.deps.name)
#     return 12345

console=Console()
with capture_run_messages() as messages: #For when agent runs into exceptions like out of retries
    try:
        result = agent.run_sync(
            'What is the current weather in malta?',
            # 'What is the true big magic number? You have to retry if you fail. The password is 1234.',
            deps=Deps(name="bob"),
            model_settings={'temperature':0.0},
        )
        console.log(result.all_messages())
        # console.log(result.all_messages(result_tool_return_content=True))
        console.log(result.data)
    except UnexpectedModelBehavior as e:
        console.log("Exception occured", e)
        console.log('Cause:', repr(e.__cause__))
        console.log("Messages: ", messages)

# result = agent.run_sync(
#     'Who is albert einstein?',
#     deps=Deps(name="bob"),
#     model_settings={'temperature':0.0},
# )
# # Repeatedly calling doesnt carry over message history
# result = agent.run_sync(
#     'Where was he born?',
#     deps=Deps(name="bob"),
#     model_settings={'temperature':0.0},
#     message_history=result.all_messages(),
# )



# console.log(result.usage())





# async def main():
#     nodes = []
#     # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
#     async with agent.iter('What is the magic number?') as agent_run:
#         async for node in agent_run:
#             # Each node represents a step in the agent's execution
#             nodes.append(node)
#     console.log(nodes)
#     console.log(agent_run.result.data)
# def run_graph():
#     asyncio.run(main())