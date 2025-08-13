import dotenv
import os
import asyncio
import httpx
import traceback
import argparse
from pathlib import Path

from rich.console import Console
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
import wandb
import weave
from openai import AsyncOpenAI

from prompts.prompts_utils import get_iteration_prompt, get_user_prompt, get_system_prompt
from tools.setup_tools import create_tools
from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics
from run_logging.wandb import setup_logging
from run_logging.log_files import log_files
from utils.create_user import create_new_user_and_rundir
from utils.dataset_utils import setup_agent_datasets
from utils.config import Config, make_config
from utils.snapshots import is_new_best, snapshot, get_new_and_best_metrics
from utils.exceptions import IterationRunFailed
from utils.openrouter_api import init_weave_with_costs
from steps.final_outcome import FinalOutcome, get_final_outcome_prompt
from steps.data_split import DataSplit, get_data_split_prompt
from steps.model_architecture import ModelArchitecture, get_model_architecture_prompt
from steps.data_representation import DataRepresentation, get_data_representation_prompt
from steps.data_exploration import DataExploration, get_data_exploration_prompt
from steps.model_training import ModelTraining, get_model_training_prompt
from feedback.feedback_agent import get_feedback, aggregate_feedback


# Load environment using shared utility
from utils.env_utils import load_environment
env = load_environment()
api_key = env['openrouter_api_key']
wandb_key = env['wandb_api_key']
proxy_url = env['proxy_url']
console = env['console']

# Create conditional weave functionality based on whether we have WANDB_API_KEY
if wandb_key:
    # Use real weave when WandB is available
    weave_op = weave.op
    weave_attributes = weave.attributes
    weave_enabled = True
else:
    # Use no-op functions when WandB is not available
    def weave_op(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    # No-op context manager for attributes
    from contextlib import nullcontext
    def weave_attributes(attrs):
        return nullcontext()
    
    weave_enabled = False

def parse_args():
    parser = argparse.ArgumentParser(description="Runs agent and outputs results to the workspace directory")
    parser.add_argument('--dataset-name', required=True, help='Name of the folder containing dataset files')
    parser.add_argument('--model', help='Openrouter-available LLM model to use', default="openai/gpt-4.1")
    parser.add_argument('--val-metric', help='Validation metric to use for the best model selection', default="ACC", choices=["AUPRC", "AUROC", "ACC", "MSE", "RMSE", "MAE", "R2"])
    parser.add_argument('--no-root-privileges', action='store_true', help='Flag to run without root privileges. This is not recommended, since it decreases security by not preventing the agent from accessing/modifying files outside of its own workspace.')
    parser.add_argument('--workspace-dir', type=Path, default=Path('../workspace/runs').resolve(), help='Path to a directory which will store the agent run and generated files')
    parser.add_argument('--prepared-datasets-dir', type=Path, default=Path('../prepared_datasets').resolve(), help='Path to a directory which contains prepared datasets.')
    parser.add_argument('--agent-datasets-dir', type=Path, default=Path('../workspace/datasets').resolve(), help='Path to a directory which contains non-test data accessible by agents.')
    parser.add_argument('--tags', nargs='*', default=[], help='(Optional) Tags for a wandb run logging')
    return parser.parse_args()

@weave_op(call_display_name=lambda call: f"Agent Run - {getattr(call.inputs.get('output_type'), '__name__', 'General') if call.inputs.get('output_type') else 'General'}")
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

@weave_op(call_display_name="ML Agent Training Pipeline")
async def main(model, feedback_model, dataset, tags, best_metric, root_privileges, workspace_dir, dataset_dir, agent_dataset_dir):
    config = make_config(model=model, 
                         feedback_model=feedback_model, 
                         dataset=dataset, 
                         tags=tags, 
                         best_metric=best_metric,
                         root_privileges=root_privileges,
                         workspace_dir=workspace_dir,
                         dataset_dir=dataset_dir,
                         agent_dataset_dir=agent_dataset_dir)

    agent_id = create_new_user_and_rundir(config)
    config.agent_id = agent_id

    # Set up W&B run first for proper integration with Weave
    wandb_run = None
    if wandb_key:
        wandb_run = setup_logging(config, api_key=wandb_key)
        print("✅ WandB logging enabled")
    else:
        print("⚠️  WANDB_API_KEY not provided - skipping experiment tracking")

    openrouter_api_key = env['openrouter_api_key']

    # Initialize Weave with cost tracking for the specific model being used
    # This should be called after wandb.init() for proper integration
    weave_client = None
    if wandb_run:  # Only initialize Weave costs if WandB is active
        weave_client = init_weave_with_costs(
            project_name=config.get_weave_project(),
            model_id=config.model,
            api_key=openrouter_api_key
        )
        print("✅ Weave cost tracking enabled")
    else:
        print("⚠️  Weave cost tracking disabled (no WandB) - skipping Weave initialization")

    async_http_client = httpx.AsyncClient(
        proxy=proxy_url if config.use_proxy else None,
        timeout= config.llm_response_timeout,
    )
    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=openrouter_api_key,
        http_client=async_http_client,
    )
    model = OpenAIModel(
        config.model,
        provider=OpenAIProvider(openai_client=client)
    )

    tools = create_tools(config)

    agent = Agent( # this is data exploration, representation, architecture reasoning agent
        model=model,
        system_prompt=get_system_prompt(config),
        tools=tools,
        model_settings={'temperature': config.temperature},
        retries=config.max_run_retries,
        result_retries=config.max_validation_retries,
    )

    split_dataset_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataSplit,
        result_retries=config.max_validation_retries,
    )

    training_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelTraining,
        result_retries=config.max_validation_retries,
    )

    validation_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= FinalOutcome,
        result_retries=config.max_validation_retries,
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
    print(f"Starting training loop with {config.iterations} iterations")
    for run_index in range(config.iterations):
        print(f"\n=== ITERATION {run_index + 1} / {config.iterations} ===")
        if run_index == 0:
            base_prompt = get_user_prompt(config)
        else:
            base_prompt = get_iteration_prompt(config, run_index, feedback)
        
        # Add Weave attributes for this iteration
        with weave_attributes({
            'iteration': run_index + 1,
            'dataset': config.dataset,
            'model': config.model,
            'temperature': config.temperature,
            'max_steps': config.max_steps,
            'best_metric': config.best_metric,
            'is_first_iteration': run_index == 0
        }):
            try:
                current_run_messages = await run_architecture(agent, validation_agent, split_dataset_agent, training_agent, config, base_prompt, iteration=run_index)
            except Exception as e:
                log_serial_metrics(prefix='validation', metrics=None, iteration=run_index)
                log_serial_metrics(prefix='train', metrics=None, iteration=run_index)
                new_metrics, best_metrics = get_new_and_best_metrics(config)
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
                log_files(config, iteration=run_index)
                continue

            try:
                print("Starting evaluation phase")
                print("  Running validation inference...")
                run_inference_and_log(config, iteration=run_index, evaluation_stage='validation')
                print("  Running training inference...")
                run_inference_and_log(config, iteration=run_index, evaluation_stage='train')
                print("  Running stealth test inference...")
                run_inference_and_log(config, iteration=run_index, evaluation_stage='stealth_test')

                new_metrics, best_metrics = get_new_and_best_metrics(config)
                all_feedbacks.append((feedback, f"Metrics after feedback incorporation: {new_metrics}", f"Best metrics so far: {best_metrics}"))
                
                # Add custom metrics to Weave call summary
                if weave_enabled:
                    try:
                        # Try to access current call context - this might vary by Weave version
                        current_call = weave.context.get_current_call() if hasattr(weave, 'context') else None
                        if current_call and hasattr(current_call, 'summary'):
                            current_call.summary.update({
                                'new_metrics': new_metrics,
                                'best_metrics': best_metrics,
                                'is_new_best': is_new_best(config),
                                'iteration': run_index + 1
                            })
                    except Exception:
                        # Silently ignore if call summary update fails
                        pass
                
                if is_new_best(config):
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

                    snapshot(config, run_index)  # Snapshotting overrides the previous snapshot, influencing the get_new_and_best_metrics function
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
                new_metrics, best_metrics = get_new_and_best_metrics(config)
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
                log_files(config, iteration=run_index)
        
    print("\nRunning final test evaluation...")
    try:
        run_inference_and_log(config, iteration=run_index, evaluation_stage='test', use_best_snapshot=True)
    except Exception as e:
        print('FINAL TEST EVAL FAIL', str(e))
        log_inference_stage_and_metrics(1)
    
    log_files(config)
    if wandb_run:
        wandb.finish()


@weave_op(call_display_name=lambda call: f"Architecture Pipeline - Iteration {call.inputs.get('iteration', 0) + 1}")
async def run_architecture(agent: Agent, validation_agent: Agent, split_dataset_agent: Agent, training_agent: Agent, config: Config, base_prompt: str, iteration: int):
    print("Step 1: Data Exploration")
    messages_data_exploration = await run_agent(
        agent=agent,
        user_prompt=base_prompt + get_data_exploration_prompt(),
        max_steps=config.max_steps,
        output_type=DataExploration, # this is overriding the output_type
        message_history=None,
    )

    if iteration == 0:
        print("Step 2: Data Split")
        messages_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config),
            max_steps=config.max_steps,
            message_history=messages_data_exploration,
        )
    else:
        print("Step 2: Data Split (skipping - not first iteration)")
        messages_split = messages_data_exploration

    print("Step 3: Data Representation")
    messages_representation = await run_agent(
        agent=agent,
        user_prompt=get_data_representation_prompt(),
        max_steps=config.max_steps,
        output_type=DataRepresentation, # this is overriding the output_type
        message_history=messages_split,
    )

    print("Step 4: Model Architecture")
    messages_architecture = await run_agent(
        agent=agent,
        user_prompt=get_model_architecture_prompt(),
        max_steps=config.max_steps,
        output_type=ModelArchitecture, # this is overriding the output_type
        message_history=messages_representation,
    )

    print("Step 5: Model Training")
    messages_training = await run_agent(
        agent=training_agent, 
        user_prompt=get_model_training_prompt(), 
        max_steps=config.max_steps,
        message_history=messages_architecture,
    )

    print("Step 6: Final Outcome & Validation")
    _messages = await run_agent(
        agent=validation_agent, 
        user_prompt=get_final_outcome_prompt(), 
        max_steps=config.max_steps,
        message_history=messages_training,
    )

    return _messages

async def run_experiment():
    args = parse_args()
    setup_agent_datasets(args.prepared_datasets_dir, args.agent_datasets_dir)

    FEEDBACK_MODEL = args.model
    await main(
        model=args.model, 
        feedback_model=FEEDBACK_MODEL, 
        dataset=args.dataset_name, 
        tags=args.tags,
        best_metric=args.val_metric, 
        root_privileges=not args.no_root_privileges, 
        workspace_dir=args.workspace_dir, 
        dataset_dir=args.prepared_datasets_dir, 
        agent_dataset_dir=args.agent_datasets_dir,
    )

if __name__ == "__main__":
    asyncio.run(run_experiment())