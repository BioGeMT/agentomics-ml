import dotenv
import os
import asyncio
import httpx
import traceback
import argparse
from pathlib import Path

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import wandb
from openai import AsyncOpenAI

from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics
from run_logging.wandb import setup_logging
from run_logging.log_files import log_files
from utils.create_user import create_new_user_and_rundir
from utils.dataset_utils import setup_nonsensitive_dataset_files_for_agent
from utils.config import Config
from utils.snapshots import is_new_best, snapshot, get_new_and_best_metrics
from utils.workspace_setup import ensure_workspace_folders
from agents.architecture import run_iteration
from utils.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.report_logger import add_metrics_to_report, add_summary_to_report, add_final_test_metrics_to_best_report, rename_and_snapshot_best_iteration_report

from feedback.feedback_agent import get_feedback, aggregate_feedback

async def main(model_name, feedback_model_name, dataset, tags, val_metric, root_privileges, workspace_dir, prepared_datasets_dir, agent_dataset_dir, iterations):
    # Initialize configuration 
    config = Config(
        model_name=model_name, 
        feedback_model_name=feedback_model_name, 
        dataset=dataset, 
        tags=tags, 
        val_metric=val_metric,
        root_privileges=root_privileges,
        workspace_dir=Path(workspace_dir),
        prepared_datasets_dir=Path(prepared_datasets_dir),
        agent_dataset_dir=Path(agent_dataset_dir),
        iterations=iterations,
    )
    ensure_workspace_folders(config)
    # Create a user for the agent
    agent_id = create_new_user_and_rundir(config)
    config.agent_id = agent_id
    config.print_summary()
    
    # initialize logging and LLMs
    dotenv.load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    wandb_key = os.getenv("WANDB_API_KEY")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    wandb_entity = os.getenv("WANDB_ENTITY")
    proxy_url = os.getenv("HTTP_PROXY")
    wandb_logged_in = setup_logging(config, api_key=wandb_key, wandb_project_name=wandb_project_name, wandb_entity=wandb_entity)

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
        config.model_name,
        provider=OpenAIProvider(openai_client=client)
    )
    feedback_model = OpenAIModel(
        config.feedback_model_name,
        provider=OpenAIProvider(openai_client=client)
    )

    await run_agentomics(config=config, default_model=model, feedback_model=feedback_model)

    if(wandb_logged_in):
        wandb.finish()

async def run_agentomics(config: Config, default_model, feedback_model):
    all_feedbacks = []
    feedback = None
    print(f"Starting training loop with {config.iterations} iterations")
    for run_index in range(config.iterations):
        print(f"\n=== ITERATION {run_index + 1} / {config.iterations} ===")
        try:
            current_run_messages = await run_iteration(config=config, model=default_model, iteration=run_index, feedback=feedback)
        except Exception as e:
            log_serial_metrics(prefix='validation', metrics=None, iteration=run_index, task_type=config.task_type)
            log_serial_metrics(prefix='train', metrics=None, iteration=run_index, task_type=config.task_type)
            new_metrics, best_metrics = get_new_and_best_metrics(config)
            all_feedbacks.append((feedback, f"Metrics after feedback incorporation: {new_metrics}", f"Best metrics so far: {best_metrics}")) # append feedback from last iteration before to process it
            feedback = await get_feedback(
                context=e.context_messages,
                extra_info=f"{e.message} {e.exception_trace}",
                config=config, 
                new_metrics=new_metrics, 
                best_metrics=best_metrics,
                is_new_best=False,
                model=feedback_model,
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
            
            if is_new_best(config):
                feedback = await get_feedback(
                    context=current_run_messages, 
                    config=config, 
                    new_metrics=new_metrics, 
                    best_metrics=best_metrics, 
                    is_new_best=True, 
                    model=feedback_model,
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
                    model=feedback_model,
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
                    model=feedback_model,
                    iteration=run_index,
                    aggregated_feedback=aggregate_feedback(all_feedbacks),
                    extra_info=f"Inference failed: {traceback.format_exc()}",
                )
        finally:
            add_metrics_to_report(config, run_index)
            await add_summary_to_report(config, run_index)
            log_files(config, iteration=run_index)
        
    print("\nRunning final test evaluation...")
    try:
        run_inference_and_log(config, iteration=None, evaluation_stage='test', use_best_snapshot=True)
        add_final_test_metrics_to_best_report(config)
    except Exception as e:
        print('FINAL TEST EVAL FAIL', str(e))
        log_inference_stage_and_metrics(1, task_type=config.task_type)

    rename_and_snapshot_best_iteration_report(config)
    log_files(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Runs agent and outputs results to the workspace directory")
    parser.add_argument('--dataset-name', required=True, help='Name of the folder containing dataset files')
    parser.add_argument('--model', help='Openrouter-available LLM model to use', default="openai/gpt-4.1")
    parser.add_argument('--no-root-privileges', action='store_true', help='Flag to run without root privileges. This is not recommended, since it decreases security by not preventing the agent from accessing/modifying files outside of its own workspace.')
    parser.add_argument('--workspace-dir', type=Path, default=Path('../workspace').resolve(), help='Path to a directory which will store agent runs, snapshots, and reports')
    parser.add_argument('--prepared-datasets-dir', type=Path, default=Path('../repository/prepared_datasets').resolve(), help='Path to a directory which contains prepared datasets.')
    parser.add_argument('--agent-datasets-dir', type=Path, default=Path('../workspace/datasets').resolve(), help='Path to a directory which contains non-test data accessible by agents.')
    parser.add_argument('--tags', nargs='*', default=[], help='(Optional) Tags for a wandb run logging')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations to run')

    val_metric_choices = get_classification_metrics_names() + get_regression_metrics_names()
    parser.add_argument('--val-metric', help='Validation metric to use for the best model selection', required=True, choices=val_metric_choices)

    return parser.parse_args()

async def run_experiment(model, dataset_name, val_metric, prepared_datasets_dir, agent_datasets_dir, workspace_dir, tags, no_root_privileges, iterations):
    setup_nonsensitive_dataset_files_for_agent(
        prepared_datasets_dir=Path(prepared_datasets_dir), 
        agent_datasets_dir=Path(agent_datasets_dir),
        dataset_name=dataset_name,
    )
    FEEDBACK_MODEL = model
    await main(
        model_name=model, 
        feedback_model_name=FEEDBACK_MODEL, 
        dataset=dataset_name, 
        tags=tags,
        val_metric=val_metric, 
        root_privileges=not no_root_privileges, 
        workspace_dir=workspace_dir, 
        prepared_datasets_dir=prepared_datasets_dir, 
        agent_dataset_dir=agent_datasets_dir,
        iterations=iterations,
    )

async def run_experiment_from_terminal():
    args = parse_args()
    await run_experiment(
        model=args.model, 
        dataset_name=args.dataset_name, 
        val_metric=args.val_metric, 
        prepared_datasets_dir=args.prepared_datasets_dir, 
        agent_datasets_dir=args.agent_datasets_dir, 
        workspace_dir=args.workspace_dir, 
        tags=args.tags,
        no_root_privileges=args.no_root_privileges,
        iterations=args.iterations,
    )

if __name__ == "__main__":
    asyncio.run(run_experiment_from_terminal())