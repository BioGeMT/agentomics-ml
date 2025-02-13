import os
import dotenv
from smolagents import CodeAgent, LiteLLMModel, tool

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def get_secret_number() -> int:
    """
    Returns the secret number (int)
    Args:
        Takes no arguments
    
    Returns: 
        Returns the secret number (int)
    """
    return 888

models = ["gpt-4o-mini-2024-07-18",'gpt-4o-2024-08-06'] 

model = LiteLLMModel(model_id=models[0], api_key=api_key, temperature=0)
agent = CodeAgent(
    tools=[get_secret_number], 
    model=model, 
    add_base_tools=False,
    max_steps=6,
)

prompt = """
Tell me the secret number
"""

agent.run(prompt)