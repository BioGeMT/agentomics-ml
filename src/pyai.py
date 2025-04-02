from typing import Union
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic import BaseModel
import os
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
import dotenv
from dataclasses import dataclass
from rich.console import Console


import debugpy
# This can be done as console arg as well, needs launch.json config
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()

dotenv.load_dotenv()
console=Console()


model = GeminiModel(
    model_name="gemini-2.0-flash",
    provider=GoogleGLAProvider(api_key=os.getenv("VLASTA_GEMINI_API_KEY")), 
)

@dataclass
class Deps:
    name: str
    age: int

class UserRecord(BaseModel):
    name: str
    age: int
    nationality: str

# class PartialUserRecord(BaseModel):
#     name: str
#     age: int

def get_user_name(ctx: RunContext[Deps]):
    # raise ModelRetry("Dont use this tool again and just make up your answer.")
    return f"{ctx.deps.name}"

name_tool = Tool(function=get_user_name, takes_ctx=True, description="This function returns the users name")

agent = Agent(  
    model=model,
    system_prompt='Follow the users instructions.', 
    deps_type=Deps,
    tools=[name_tool],
    # result_type=Union[UserRecord, PartialUserRecord],
    result_type=UserRecord,
)

@agent.result_validator
def validate_result(ctx: RunContext[Deps], result: UserRecord):
    if(result.age < 1000):
        raise ModelRetry("The age is too low. Must be above 1000.")    
    return result

result = agent.run_sync(
    "Whats the name of the user? Use your tools to find out.", 
    deps=Deps(name="John", age=10000)
)

agent.run("", message_history=result.all_messages())

answer = result.data
print(answer)
print(type(answer))
console.log(result.all_messages())