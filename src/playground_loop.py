import dotenv
import os
import asyncio
import httpx
import traceback

from rich.console import Console
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
import wandb
from openai import AsyncOpenAI
from timeout_function_decorator import timeout


from prompts.prompts_utils import get_iteration_prompt, get_user_prompt, get_system_prompt
from tools.bash import create_bash_tool
from tools.write_python_tool import create_write_python_tool
from tools.run_python_tool import create_run_python_tool
from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics
from run_logging.wandb import setup_logging
from run_logging.log_files import log_files
from utils.create_user import create_new_user_and_rundir
from utils.models import MODELS
from utils.snapshots import is_new_best, snapshot, get_new_and_best_metrics
from utils.exceptions import IterationRunFailed
from utils.api_keys import create_new_api_key, get_api_key_usage, delete_api_key
from steps.final_outcome import FinalOutcome, get_final_outcome_prompt
from steps.data_split import DataSplit, get_data_split_prompt
from steps.model_architecture import ModelArchitecture, get_model_architecture_prompt
from steps.data_representation import DataRepresentation, get_data_representation_prompt
from steps.data_exploration import DataExploration, get_data_exploration_prompt
from steps.model_training import ModelTraining, get_model_training_prompt
from feedback.feedback_agent import get_feedback, aggregate_feedback

dotenv.load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
wandb_key = os.getenv("WANDB_API_KEY")
proxy_url = os.getenv("HTTP_PROXY")
console = Console()

async def run_agent(agent: Agent, user_prompt: str, max_steps: int, message_history: list | None, output_type: BaseModel = None):
    with capture_run_messages() as messages:
        try:
            result = await agent.run(
                user_prompt=user_prompt,
                usage_limits=UsageLimits(request_limit=max_steps),
                output_type=output_type,
                message_history=message_history,
            )
            console.log(result.new_messages())
            return result.all_messages()
        except Exception as e:
            trace = traceback.format_exc()
            print('Agent run failed', trace)
            raise IterationRunFailed(
                message="Run didnt finish properly", 
                context_messages=messages,
                exception_trace=trace,
            )

async def main(model, feedback_model, dataset, tags, best_metric):
    agent_id = create_new_user_and_rundir()
    config = {
        "agent_id": agent_id,
        "model": model,
        "feedback_model": feedback_model,
        "temperature": 1,
        "max_steps": 100, #TODO rename, this is per-step limit
        "max_run_retries": 1,
        "max_validation_retries": 5,
        "tags": tags,
        "dataset": dataset,
        # "prompt": "BioPrompt_v1.yaml", #TODO cleanup, not used
        "use_proxy" : True,
        "best_metric" : best_metric, #TODO rename into validation_metric
        "iterations": 5,
        "llm_response_timeout": 60* 15,
        "bash_tool_timeout": 60 * 5, 
        "run_python_tool_timeout": 60 * 60 * 6, #This affects max training time
        "credit_budget": 30,
        "max_tool_retries": 5,
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
        api_key=api_key,
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
            max_retries=config['max_tool_retries'],
            proxy=config['use_proxy'],
            conda_prefix=True,
            auto_torch=False),
        create_write_python_tool( 
            agent_id=config['agent_id'], 
            max_retries=config['max_tool_retries'],),
        create_run_python_tool( #Tries to create the same-name conda environment
            agent_id=config['agent_id'],
            timeout=config['run_python_tool_timeout'],
            proxy=config['use_proxy'],
            max_retries=config['max_tool_retries'],),
    ]

    agent = Agent( # this is data exploration, representation, architecture reasoning agent
        model=model,
        system_prompt=get_system_prompt(config),
        tools=tools,
        model_settings={'temperature': config['temperature']},
        retries=config["max_run_retries"],
        result_retries=config["max_validation_retries"],
    )

    split_dataset_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config['temperature']},
        output_type=DataSplit,
        result_retries=config["max_validation_retries"],
    )

    training_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config['temperature']},
        output_type=ModelTraining,
        result_retries=config["max_validation_retries"],
    )

    validation_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config['temperature']},
        output_type= FinalOutcome,
        result_retries=config["max_validation_retries"],
    )
    @split_dataset_agent.output_validator
    async def validate_split_dataset(result: DataSplit) -> DataSplit:
        if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
            raise ModelRetry("Split dataset files do not exist.")
        return result
    
    @training_agent.output_validator
    async def validate_training(result: ModelTraining) -> ModelTraining:
        if not os.path.exists(result.path_to_train_file):
            raise ModelRetry("Train file does not exist.")
        if not os.path.exists(result.path_to_model_file):
            raise ModelRetry("Model file does not exist.")
        return result

    @validation_agent.output_validator
    async def validate_inference(result: FinalOutcome) -> FinalOutcome:
        if not os.path.exists(result.path_to_inference_file):
            raise ModelRetry("Inference file does not exist.")
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run') 
        return result       
    
    all_feedbacks = []
    feedback = None
    for run_index in range(config["iterations"]):
        if run_index == 0:
            base_prompt = get_user_prompt(config)
        else:
            base_prompt = get_iteration_prompt(config, run_index, feedback)
        try:
            current_run_messages = await run_architecture(agent, validation_agent, split_dataset_agent, training_agent, config, base_prompt, iteration=run_index)
        except Exception as e:
            stats = get_api_key_usage(openrouter_api_key_hash)
            if stats['usage'] >= stats['limit']:
                wandb.log(stats)
                wandb.log({"out_of_credits": True})
                print('RAN OUT OF CREDITS')
                log_files(config['agent_id'], iteration=run_index)
                break #Break looping and go to test evaluation of the best model
            
            log_serial_metrics(prefix='validation', metrics=None, iteration=run_index)
            log_serial_metrics(prefix='train', metrics=None, iteration=run_index)
            new_metrics, best_metrics = get_new_and_best_metrics(config['agent_id'])
            all_feedbacks.append((feedback, f"Metrics after feedback incorporation: {new_metrics}", f"Best metrics so far: {best_metrics}")) # append feedback from last iteration before to process it
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
            log_files(config['agent_id'], iteration=run_index)
            continue

        try:
            run_inference_and_log(config, iteration=run_index, evaluation_stage='validation')
            run_inference_and_log(config, iteration=run_index, evaluation_stage='train')
            run_inference_and_log(config, iteration=run_index, evaluation_stage='stealth_test')

            new_metrics, best_metrics = get_new_and_best_metrics(config['agent_id'])
            all_feedbacks.append((feedback, f"Metrics after feedback incorporation: {new_metrics}", f"Best metrics so far: {best_metrics}"))
            if is_new_best(config['agent_id'], config['best_metric']):
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
                feedback =await get_feedback(
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
            new_metrics, best_metrics = get_new_and_best_metrics(config['agent_id'])
            all_feedbacks.append((feedback, f"Metrics after feedback incorporation: {new_metrics}", f"Best metrics so far after the feedback incorporation: {best_metrics}"))
            feedback = await get_feedback(
                    current_run_messages, 
                    config, 
                    new_metrics, 
                    best_metrics, 
                    is_new_best=False, 
                    api_key=openrouter_api_key, 
                    iteration=run_index,
                    aggregated_feedback=aggregate_feedback(all_feedbacks),
                    extra_info=f"Inference failed: {traceback.format_exc()}",
                    )
            print(feedback)
        finally:
            log_files(config['agent_id'], iteration=run_index)
            stats = get_api_key_usage(openrouter_api_key_hash)
            wandb.log({f"iteration_usage": stats['usage']})
        
    stats = get_api_key_usage(openrouter_api_key_hash)
    wandb.log(stats)
    
    try:
        run_inference_and_log(config, iteration=run_index, evaluation_stage='test', use_best_snapshot=True)
    except Exception as e:
        print('FINAL TEST EVAL FAIL', str(e))
        log_inference_stage_and_metrics(1)
    
    log_files(config['agent_id'])
    delete_api_key(openrouter_api_key_hash)
    wandb.finish()


async def run_architecture(agent: Agent, validation_agent: Agent, split_dataset_agent: Agent, training_agent: Agent, config: dict, base_prompt: str, iteration: int):
    messages_data_exploration = await run_agent(
        agent=agent,
        user_prompt=base_prompt + get_data_exploration_prompt(),
        max_steps=config["max_steps"],
        output_type=DataExploration, # this is overriding the output_type
        message_history=None,
    )

    if iteration == 0:
        messages_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config),
            max_steps=config["max_steps"],
            message_history=messages_data_exploration,
        )
    else:
        messages_split = messages_data_exploration

    messages_representation = await run_agent(
        agent=agent,
        user_prompt=get_data_representation_prompt(),
        max_steps=config["max_steps"],
        output_type=DataRepresentation, # this is overriding the output_type
        message_history=messages_split,
    )

    messages_architecture = await run_agent(
        agent=agent,
        user_prompt=get_model_architecture_prompt(),
        max_steps=config["max_steps"],
        output_type=ModelArchitecture, # this is overriding the output_type
        message_history=messages_representation,
    )

    messages_training = await run_agent(
        agent=training_agent, 
        user_prompt=get_model_training_prompt(), 
        max_steps=config["max_steps"],
        message_history=messages_architecture,
    )

    _messages = await run_agent(
        agent=validation_agent, 
        user_prompt=get_final_outcome_prompt(), 
        max_steps=config["max_steps"],
        message_history=messages_training,
    )

    return _messages


async def run_experiments():
    best_metrics = {
        "human_nontata_promoters": "ACC",
        "human_enhancers_cohn": "ACC",
        "drosophila_enhancers_stark": "ACC",
        "human_enhancers_ensembl": "ACC",
        "human_ocr_ensembl": "ACC",
        "AGO2_CLASH_Hejret2023": "AUPRC",
    }
    DATASETS=["human_nontata_promoters"]#,"human_enhancers_cohn","drosophila_enhancers_stark","human_enhancers_ensembl","AGO2_CLASH_Hejret2023","human_ocr_ensembl"]
    MODELS_TO_RUN = [MODELS.GPT4_1]
    TAGS = ["test"]
    for dataset in DATASETS:
        for model in MODELS_TO_RUN:
            FEEDBACK_MODEL=model
            await main(model, FEEDBACK_MODEL, dataset, TAGS, best_metrics[dataset])


if __name__ == "__main__":
    asyncio.run(run_experiments())