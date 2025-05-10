import dotenv
import os
import asyncio
import httpx
import traceback

from rich.console import Console
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
import wandb
from openai import AsyncOpenAI
from timeout_function_decorator import timeout


from prompts.prompts_utils import load_prompts
from tools.bash import create_bash_tool
from tools.write_python_tool import create_write_python_tool
from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics
from run_logging.wandb import setup_logging
from run_logging.log_files import log_files
from utils.create_user import create_new_user_and_rundir
from utils.models import MODELS
from utils.exceptions import IterationRunFailed
from utils.snapshots import is_new_best, snapshot, get_new_and_best_metrics
from utils.api_keys import create_new_api_key, get_api_key_usage, delete_api_key
from steps.final_outcome import FinalOutcome
from steps.data_split import DataSplit
from steps.model_architecture import ModelArchitecture
from steps.data_representation import DataRepresentation
from steps.data_exploration import DataExploration
from feedback.feedback_agent import get_feedback, aggregate_feedback

dotenv.load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
wandb_key = os.getenv("WANDB_API_KEY")
proxy_url = os.getenv("HTTP_PROXY")
console = Console()

async def run_agent(agent: Agent, user_prompt: str, max_steps: int, result_type: BaseModel, message_history: list | None):
    """
    Executes the agent with the given prompt and returns the structured result or exception.
    """
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
        # except UnexpectedModelBehavior as e: #Validation retry runout is this exception
        #     raise e
        except Exception as e:
            raise IterationRunFailed(
                message="Run didnt finish properly", 
                context_messages=messages,
                exception_trace=traceback.format_exc()
            )

async def main():

    agent_id = create_new_user_and_rundir()
    config = {
        "agent_id": agent_id,
        "model": MODELS.GPT4_1_mini,
        "feedback_model": MODELS.GPT4_1_mini,
        "temperature": 1,
        "max_steps": 30, #TODO rename, this is per-step limit
        "max_run_retries": 1,
        "max_validation_retries": 5,
        "tags": ["double_run"],
        "dataset": "human_nontata_promoters",
        "prompt": "BioPrompt_v1.yaml",
        "use_proxy" : True,
        "best_metric" : "ACC", #TODO rename into validation_metric
        "iterations": 2,
        "llm_response_timeout": 60* 15,
        "bash_tool_timeout": 60 * 15, #TODO this is also max-training time, increase
        "write_python_tool_timeout": 60 * 5,
        "credit_budget": 10
    }
    setup_logging(config, api_key=wandb_key)

    api_key_data = create_new_api_key(name=config["agent_id"], limit=config["credit_budget"])
    openrouter_api_key = api_key_data['key']
    openrouter_api_key_hash = api_key_data['hash']

    async_http_client = httpx.AsyncClient(
            proxy=proxy_url if config["use_proxy"] else None,
            timeout= config["llm_response_timeout"],
        )
    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=openrouter_api_key,
        http_client=async_http_client,
    )
    model = OpenAIModel(
        config['model'],
        provider=OpenAIProvider(openai_client=client)
    )

    tools =[
        create_bash_tool(
            agent_id=config['agent_id'],
            timeout=config['bash_tool_timeout'], 
            autoconda=True,
            max_retries=1,
            proxy=config['use_proxy'],
            conda_prefix=True,
            auto_torch=False),
        create_write_python_tool(
            agent_id=config['agent_id'], 
            timeout=config['write_python_tool_timeout'], 
            add_code_to_response=False,
            max_retries=1),
    ]

    agent = Agent( # this is data exploration, representation, architecture reasoning agent
        model=model,
        system_prompt=load_prompts(config["prompt"])["system_prompt"],
        tools=tools,
        model_settings={'temperature': config['temperature']},
        retries=config["max_run_retries"],
        result_retries=config["max_validation_retries"],
    )

    split_dataset_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config['temperature']},
        result_type=DataSplit,
        result_retries=config["max_validation_retries"],
    )

    validation_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config['temperature']},
        result_type= FinalOutcome,
        result_retries=config["max_validation_retries"],
    )
    @split_dataset_agent.result_validator
    def validate_split_dataset(result: DataSplit) -> DataSplit:
        if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
            raise ModelRetry("Split dataset files do not exist.")
        return result

    @validation_agent.result_validator
    async def validate_inference(result: FinalOutcome) -> FinalOutcome:
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run') 
        return result       
    
    all_feedbacks = []
    feedback = None
    for run_index in range(config["iterations"]):
        if run_index == 0:
            base_prompt = f"""
                You are using a linux system.
                You have access to both CPU and GPU resources.
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
                Create a RandomForestClassifier model using sklearn.
                Run all bash commands and python command in a way that prints the least possible amount of tokens into the console.
                Create the best possible classifier that will generalize to new unseen data, explore your data before building the model.
                Note that you can use GPU-accelerated libraries such as PyTorch.
            """
        else:
            base_prompt = f"""
                Here is the feedback the previous run:
                {feedback}
                Ensure that there is an 'inference.py' script at /workspace/runs/{config['agent_id']}/inference.py.
                - If the previous run failed to create this script, generate it from scratch following the same input/output specs.
                - If the script exists, inspect its contents, describe what improvements you'd make, then implement those changes.
                Use the existing context to guide your refinements.
                The list of files in the workspace (/workspace/runs/{config['agent_id']}/) is:
                Change the model to a simple DecisionTreeClassifier.
            """

        try:
            current_run_messages = await run_architecture(agent, validation_agent, split_dataset_agent, config, base_prompt, iteration=run_index)
        except Exception as e:
            stats = get_api_key_usage(openrouter_api_key_hash)
            if stats['usage'] >= stats['limit']:
                wandb.log(stats)
                wandb.log({"out_of_credits": True})
                print('RAN OUT OF CREDITS')
                log_files(config['agent_id'], run_index)
                raise e #Kill the run
            
            # if(isinstance(e, UnexpectedModelBehavior)):
            #     stats = get_api_key_usage(openrouter_api_key_hash)
            #     wandb.log(stats)
            #     delete_api_key(openrouter_api_key_hash)
            #     if stats['usage'] >= stats['limit']:
            #         wandb.log({"out_of_credits": True})
            #     print('FAIL DURING ARCHITECTURE RUN')
            #     print({traceback.format_exc()})
            #     log_files(config['agent_id'], run_index)
            #     raise e #Kill the run
            
            log_serial_metrics(prefix='validation', metrics=None, iteration=run_index)
            log_serial_metrics(prefix='train', metrics=None, iteration=run_index)
            new_metrics, best_metrics = get_new_and_best_metrics(config['agent_id'])
            all_feedbacks.append(feedback)
            feedback = await get_feedback(
                context=e.context_messages,
                extra_info=f"{e.message} {e.exception_trace}",
                config=config, 
                new_metrics=new_metrics, 
                best_metrics=best_metrics,
                is_new_best=False,
                api_key=openrouter_api_key,
                aggregated_feedback=aggregate_feedback(all_feedbacks),
                iteration=run_index
            )
            log_files(config['agent_id'], run_index)
            continue

        try:
            run_inference_and_log(config, iteration=run_index, evaluation_stage='validation')
            run_inference_and_log(config, iteration=run_index, evaluation_stage='train')

            # TODO aggregate feedback over all iterations
            # TODO we dont need feedback for the last iteration
            all_feedbacks.append(feedback)
            if is_new_best(config['agent_id'], config['best_metric']):
                new_metrics, best_metrics = get_new_and_best_metrics(config['agent_id'])
                feedback = await get_feedback(
                    context=current_run_messages, 
                    config=config, 
                    new_metrics=new_metrics, 
                    best_metrics=best_metrics, 
                    is_new_best=True, 
                    api_key=openrouter_api_key, 
                    iteration=run_index,
                    aggregated_feedback=aggregate_feedback(all_feedbacks)
                )

                snapshot(config['agent_id'], run_index)  # Snapshotting overrides the previous snapshot, influencing the get_new_and_best_metrics function
            else:
                feedback = await get_feedback(
                    current_run_messages, 
                    config, 
                    new_metrics, 
                    best_metrics, 
                    is_new_best=False, 
                    api_key=openrouter_api_key, 
                    iteration=run_index,
                    aggregated_feedback=aggregate_feedback(all_feedbacks)
                    )

        except Exception as e:
            # If validation fails on last (or all) itertation, we fail on last test as well - should we catch the exception and just say we dont have anything successful?
            feedback = f'VALIDATION EVAL FAIL: {traceback.format_exc()}'
            #TODO call feedback agent with the exception
            all_feedbacks.append(feedback)
            print(feedback)
        finally:
            log_files(config['agent_id'], run_index)
            stats = get_api_key_usage(openrouter_api_key_hash)
            wandb.log({f"iteration_usage": stats['usage']})
        
    stats = get_api_key_usage(openrouter_api_key_hash)
    wandb.log(stats)
    run_inference_and_log(config, iteration=run_index, evaluation_stage='test', use_best_snapshot=True)
    log_files(config['agent_id'])
    delete_api_key(openrouter_api_key_hash)
    wandb.finish()

async def run_architecture(agent: Agent, validation_agent: Agent, split_dataset_agent: Agent, config: dict, base_prompt: str, iteration: int):
    messages_data_exploration = await run_agent(
        agent=agent,
        user_prompt=base_prompt + "\nYour first task: explore the dataset.",
        max_steps=config["max_steps"],
        result_type=DataExploration, # this is overriding the result_type
        message_history=None,
    )

    if iteration == 0:
        split_prompt = f"""
            Split the training dataset into training and validation sets:
            - Save 'train.csv' and 'validation.csv' in /workspace/runs/{config['agent_id']}/.
            Return the absolute paths to these files.
        """
        messages_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=split_prompt,
            max_steps=config["max_steps"],
            result_type=None,
            message_history=messages_data_exploration,
        )
    else:
        messages_split = messages_data_exploration

    messages_representation = await run_agent(
        agent=agent,
        user_prompt="Next task: define the data representation. (Do not implement yet.)",
        max_steps=config["max_steps"],
        result_type=DataRepresentation, # this is overriding the result_type
        message_history=messages_split,
    )

    messages_architecture = await run_agent(
        agent=agent,
        user_prompt="Next task: choose the model architecture. (Do not implement yet.)",
        max_steps=config["max_steps"],
        result_type=ModelArchitecture, # this is overriding the result_type
        message_history=messages_representation,
    )

    _messages = await run_agent(
        agent=validation_agent, 
        user_prompt="Continue with your task", 
        max_steps=config["max_steps"],
        result_type=None,
        message_history=messages_architecture,
    )

    return _messages

if __name__ == "__main__":
    try:
        time_budget_in_hours = 1 
        asyncio.run(timeout(60*60*time_budget_in_hours)(main)()) #TODO parametrize timeout
    except TimeoutError as e:
        log_inference_stage_and_metrics(0)
        wandb.log({"timed_out": True}) #TODO log usage until the timeout
        wandb.finish()