import os
import dotenv
from smolagents import CodeAgent, LiteLLMModel, tool
from telemetry_setup import logging_setup

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
logging_setup()

models = ["gpt-4o-mini-2024-07-18",'gpt-4o-2024-08-06'] 

model = LiteLLMModel(model_id=models[0], api_key=api_key, temperature=0)
agent = CodeAgent(
    tools=[], 
    model=model, 
    add_base_tools=False,
    additional_authorized_imports=["*"],
    max_steps=5,
)

agent.run('What is sqrt of 5**5')