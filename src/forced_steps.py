import dotenv
import os
import asyncio
import httpx

from rich.console import Console
from pydantic import Field, BaseModel, field_validator
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, RunContext, UsageLimitExceeded, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
import wandb

from steps.data_exploration import DataExploration
from steps.data_representation import DataRepresentation
from steps.model_architecture import ModelArchitecture
from prompts.prompts_utils import load_prompts
from run_logging.evaluate_log_run import evaluate_log_run
from run_logging.wandb import setup_logging
from tools.bash import create_bash_tool
from tools.write_python_tool import create_write_python_tool
from run_logging.evaluate_log_run import dry_run_evaluate_log_run
from utils.create_user import create_new_user_and_rundir
from utils.models import MODELS

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
wandb_key = os.getenv("WANDB_API_KEY")
proxy = os.getenv("HTTP_PROXY")
console = Console() # rich console for pretty printing

async_http_client = httpx.AsyncClient(
    proxy=proxy,
    timeout= 1 * 60
)

async def run_agent(agent: Agent, user_prompt, max_steps, result_type, message_history):
    with capture_run_messages() as messages:
        try:
            async with agent.iter(
                    user_prompt=user_prompt,
                    usage_limits=UsageLimits(request_limit=max_steps),
                    result_type=result_type,
                    message_history=message_history,
                ) as agent_run:
                async for node in agent_run:
                    console.log(node)
                return agent_run.result.all_messages()
        except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
            console.log("Exception occured", e)
            console.log('Cause:', repr(e.__cause__))
            console.log("Messages: ", messages)
            return None

async def run_architecture(agent, config):
    start_user_prompt=f"""
    You are using a linux system.
    You have access to CPU only, no GPU.
    You can only work in /workspace/runs/{config['agent_id']} directory.
    You are provided with your own already activated environment called {config['agent_id']}_env .
    Use this environment to install any packages you need (use non-verbose mode for installations, run conda installations with -y option).
    Write all your python scripts in files. 
    Run all scripts in this environment.
    Your final output should be inference.py script, located in the workspace folder (/workspace/runs/{config['agent_id']}/inference.py). 
    This script will be separated from the training script and only load the already trained model and run inference.
    Make sure the model file exists before you submit your solution!
    Any path inside the inference.py script should be absolute (eg. /workspace/runs/{config['agent_id']}/model.pth).
    The script will be taking the following named arguments:
        --input (an input file path, file is of the same format as your training data (except the label column))
        --output (the output file path, this file should be a one column csv file with the predictions, the column name should be 'prediction')
    Look into the workspace/datasets/{config['dataset']} folder and create a ML classifier using files in there.
    Use the ..._train.csv as training data.
    Run all bash commands and python command in a way that prints the least possible amount of tokens into the console.
    Create the best possible classifier that will generalize to new unseen data, explore your data before building the model.
    Note that you can use GPU-accelerated libraries such as PyTorch.
    """
    
    messages_data_exploration = await run_agent(
        agent=agent, 
        user_prompt=start_user_prompt + "Your first task is to explore the data.", 
        max_steps=config["max_steps"],
        result_type=DataExploration,
        message_history=None,
    )

    messages_representation_step = await run_agent(
        agent=agent, 
        user_prompt="Your next task is to choose the data representation. (Don't implement it yet)", 
        max_steps=config["max_steps"],
        result_type=DataRepresentation,
        message_history=messages_data_exploration,
        # deps=Deps(name="bob")
    )

    messages_architecture_step = await run_agent(
        agent=agent, 
        user_prompt="Your next task is to choose the model architecture. (Don't implement it yet)", 
        max_steps=config["max_steps"],
        result_type=ModelArchitecture,
        message_history=messages_representation_step,
        # deps=Deps(name="bob")
    )

    #We can validate like this, or with result_validator but then we'd need separate agents
    class FinalOutcome(BaseModel):
        path_to_inference_file: str = Field(description="Absolute path to the inference.py file")

        @field_validator('path_to_inference_file')  
        @classmethod
        def validate_inference(cls, value):
            eval_output = dry_run_evaluate_log_run(config)
            console.log("Validating inference script")
            console.log(eval_output)

            if eval_output.returncode != 0:
                raise ModelRetry(str(eval_output))
            return value
        
    _messages = await run_agent(
        agent=agent, 
        user_prompt="Continue with your task", 
        max_steps=config["max_steps"],
        result_type=FinalOutcome,
        message_history=messages_architecture_step,
    )

async def main():
    for _ in range(1):
        agent_id = create_new_user_and_rundir()

        config = {
            "agent_id" : agent_id,
            "model" : MODELS.GPT4o,
            "temperature" : 1,
            "max_steps" : 30,
            "max_run_retries" : 1,
            "max_validation_retries" : 5,
            "tags" : ["testing"],
            "dataset" : "human_non_tata_promoters",
            "prompt" : "toolcalling_agent.yaml",
        }

        setup_logging(config, api_key=wandb_key)

        model = OpenAIModel( #Works for openrouter as well
            config['model'],
            provider=OpenAIProvider(api_key=api_key,
                                    http_client=async_http_client)
        )

        agent = Agent(
            model=model,
            system_prompt= load_prompts(config["prompt"])["system_prompt"],
            tools =[
                create_bash_tool(
                    agent_id=config['agent_id'],
                    timeout=60 * 15, 
                    autoconda=True,
                    max_retries=1,
                    proxy=True,
                    auto_torch=True),
                create_write_python_tool(
                    agent_id=config['agent_id'], 
                    timeout=60 * 5, 
                    add_code_to_response=False,
                    max_retries=1),
            ],
            model_settings={'temperature':config['temperature']},
            retries=config["max_run_retries"],
            result_retries=config["max_validation_retries"],
        )

        await run_architecture(agent, config)
        evaluate_log_run(config)
        wandb.finish()

asyncio.run(main())
