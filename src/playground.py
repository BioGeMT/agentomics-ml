import os
import dotenv
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent
from tools.bash import bash_tool
import wandb

dotenv.load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
api_key = os.getenv("OPENAI_API_KEY")

model = "gpt-4o-mini-2024-07-18"
wandb.init(
    entity="ceitec-ai",
    project="BioAgents",
    tags=["testing"],
    config={
        "model":model,
    }
)

model = LiteLLMModel(
    model_id=model, 
    api_key=api_key, 
    temperature=1,
)

agent = ToolCallingAgent(
    tools=[bash_tool], 
    model=model, 
    add_base_tools=False,
    max_steps=5,
)

agent.run("What packages are installed in the multiagent-ml-env conda environment?")

wandb.log(
    {
        "Agent F1 score": 0.5087987, 
    }
)

