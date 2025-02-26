import os
import dotenv
from smolagents import LiteLLMModel, ToolCallingAgent
from tools.bash import bash_tool
from models import MODELS
from run_logging.wandb import setup_logging
from run_logging.evaluate_log_run import evaluate_log_run

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") #Openrouter models need their OPENROUTER API key
wandb_key = os.getenv("WANDB_API_KEY")

tools = [bash_tool]
config = {
    "agent_id" : 'placeholder_0',
    "model" : MODELS.GPT4o,
    "temperature" : 1,
    "max_steps" : 30,
    "dataset" : "human_non_tata_promoters",
    "tags" : ["testing"],
    "tools" : [tool.name for tool in tools],
}

setup_logging(config, api_key=wandb_key)

model = LiteLLMModel(
    model_id=config['model'], 
    api_key=api_key, 
    temperature=config["temperature"],
)

agent = ToolCallingAgent(
    tools=tools, 
    model=model, 
    add_base_tools=False,
    max_steps=config["max_steps"],
)

#TODO add this to the system prompt
prompt_base = f"""
You are using a linux system.
You have access to CPU only, no GPU.
Before you do anything, run cd /workspace (this is the only directory that is yours to work in).
Create a new conda environment called {config['agent_id']}_env (run the installation in non-verbose mode). 
Use this environment to install any packages you need (use non-verbose mode for installations).
Write all your python scripts in files. 
Run all scripts in this environment.
Your final output should be inference.py script, located in the workspace folder (/workspace/inference.py). 
This script will be separated from the training script and only load the already trained model and run inference.
Any path inside the inference.py script should be absolute (eg. /workspace/model.pth).
The script will be taking the following named arguments:
    --input (an input file path, file is of the same format as your training data (except the label column))
    --output (the output file path, this file should be a one column csv file with the predictions, the column name should be 'prediction')

Create a prototype model, with only few training steps, no need to optimize it for performance.
"""
user_prompt = """
Look into the workspace/datasets/human_non_tata_promoters folder and create a ML classifier using files in there.
Use the ..._train.csv as training data.
Use the ..._test.csv as testing data.
Don't ever print the whole dataset or a big part of the dataset, it will clog up your context and prevent you from continuing your task.
Any one script can't run for more than 15 minutes.

Column descriptions:
sequence: DNA sequence to be classified, alphabet: A, C, G, T, N
Make sure to tokenize for all characters from the alphabet.

class: 1 if the promoter is a non-TATA promoter, 0 otherwise
"""

agent.run(prompt_base + user_prompt)
evaluate_log_run(config)