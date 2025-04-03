from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import dotenv
import os
from utils.create_user import create_new_user_and_rundir
# import logfire
from run_logging.evaluate_log_run import evaluate_log_run
from pydantic_ai.providers.openai import OpenAIProvider
from prompts.prompts_utils import load_prompts
import wandb
from run_logging.wandb import setup_logging
from rich.console import Console
from run_logging.memory_logging import print_data

# logfire.configure(data_dir='/logfire/.logfire')
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
wandb_key = os.getenv("WANDB_API_KEY")

console = Console() # rich console for pretty printing

agent_id = create_new_user_and_rundir()

config = {
    "agent_id" : agent_id,
    "tags" : ["test_pydantic_ai"],
    "dataset" : "human_non_tata_promoters",
    "prompt" : "toolcalling_agent.yaml",
}

setup_logging(config, api_key=wandb_key)


model = OpenAIModel( #Works for openrouter as well
    "gpt-4o-2024-08-06", 
    # 'deepseek/deepseek-r1','
    # "meta-llama/llama-3.3-70b-instruct",
    # base_url='https://openrouter.ai/api/v1',
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    provider=OpenAIProvider(api_key=api_key)
)

basic_agent = Agent(
    model=model, 
    system_prompt= load_prompts(config["prompt"]),
    # tools=[bash_tool],
)

timeout = 60
from tools.bash_helpers import BashProcess
bash = BashProcess(
    agent_id=agent_id,
    strip_newlines = False,
    return_err_output = True,
    persistent = True, # cd will change it for the next command etc... (better for the agent)
    timeout = timeout, #Seconds to wait for a command to finish
)

@basic_agent.tool_plain
def _bash(command):
    """
    A persistent bash. 
    Use this to execute bash commands. 
    Input should be a valid bash command.

    Examples:
    - "ls"
    - "cd /workspace"
    - "mkdir test"
    - "echo \\"hello world\\" > test.txt"
    - "conda create -n my_env python=3.8 matplotlib -c conda-forge -y"
    - "source activate my_env"
    - "echo -e \\"import numpy as np\nx = np.linspace(0, 10, 100)\nprint('data:',x)\\" > /workspace/numpy_test.py"
        -  wrap your python code in double quotes prefixed with a backslash (\") to allow for correct bash interpretation.
        -  to print in python, never use double quotes (") as they will be interpreted by bash, use only single quotes (') instead.
    - "python /workspace/numpy_test.py"

    Args:
        command (str): A valid bash command.
    """
    return bash.run(command)

print(config)
user_prompt = f"""
You are using a linux system.
You have access to CPU only, no GPU.
You can only work in /workspace/runs/{config['agent_id']} directory.
Create a new conda environment called {config['agent_id']}_env (run the installation in non-verbose mode with -q). 
Use this environment to install any packages you need (use non-verbose mode for installations).
Write all your python scripts in files. 
Run all scripts in this environment.
Your final output should be inference.py script, located in the workspace folder (/workspace/runs/{config['agent_id']}/inference.py). 
This script will be separated from the training script and only load the already trained model and run inference.
Any path inside the inference.py script should be absolute (eg. /workspace/runs/{config['agent_id']}/model.pth).
The script will be taking the following named arguments:
    --input (an input file path, file is of the same format as your training data (except the label column))
    --output (the output file path, this file should be a one column csv file with the predictions, the column name should be 'prediction')


Look into the workspace/datasets/{config['dataset']} folder and create a ML classifier using files in there.
Use the ..._train.csv as training data.
Use the ..._test.csv as testing data.
Run all bash commands and python command in a way that prints the least possible amount of tokens into the console.
Create the best possible classifier that will generalize to new unseen data.


Column descriptions:
sequence: DNA sequence to be classified, alphabet: A, C, G, T, N
Make sure to tokenize for all characters from the alphabet.

class: 1 if the promoter is a non-TATA promoter, 0 otherwise

Validate your intermediate steps created files that are needed for the final inference script to run successfully before you run your final answer.
Also make sure the inference script exists in your folder and is runnable, it will be subsequently called and you will be evaluated based on the model's generalization performance on hidden test data.
"""
message_history = basic_agent.run_sync(user_prompt=user_prompt)

evaluate_log_run(config)

print_data(message_history.data, console=console)
print_data(message_history.all_messages(), console=console)

wandb.finish()
