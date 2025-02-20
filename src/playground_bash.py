import os
import dotenv
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent
from telemetry_setup import logging_setup
from tools.bash import bash_tool

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
logging_setup()

model = LiteLLMModel(
    model_id="gpt-4o-mini-2024-07-18", 
    api_key=api_key, 
    temperature=1,
)
agent = ToolCallingAgent(
    tools=[bash_tool], 
    model=model, 
    add_base_tools=False,
    max_steps=30,
)

#TODO add to system prompt
prompt_base = """
You are using a linux system.
You have access to CPU only, no GPU.
Before you do anything, run cd /workspace (this is the only directory that is yours to modify).
Create a new conda environment called agent_env (run the installation in non-verbose mode). 
Use this environment to install any packages you need (use non-verbose mode for installations).
Write all your python scripts in files. 
Run all scripts in this environment.
"""
user_prompt = """
Look into the workspace/datasets/human_non_tata_promoters folder and create a ML classifier using files in there.
Use the _train.csv as training data.
Use the _test.csv as testing data.
Let me know the model's accuracy on this test data.
Don't ever print the whole dataset or a big part of the dataset, it will clog up your context and prevent you from continuing your task.
Any one script can't run for more than 15 minutes.

There is a file called dataset_description.md that has more information helpful for this task. Feel free to use it.

Column descriptions:
sequence: DNA sequence to be classified {alphabet: A, C, G, T, N}

class: 1 if the promoter is a non-TATA promoter, 0 otherwise
"""
# Provide an inference script that is runnable from console.
# Try to overcome 86% accuracy on the _test data.

agent.run(prompt_base + user_prompt)

