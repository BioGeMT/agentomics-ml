import asyncio
import traceback
import argparse
from pathlib import Path
import os

import wandb

from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_serial_metrics, log_feedback_failure
from run_logging.wandb_setup import setup_logging
from run_logging.log_files import log_files, export_config_to_workspace
from utils.env_utils import are_wandb_vars_available
from utils.create_user import create_run_and_snapshot_dirs
from utils.dataset_utils import setup_nonsensitive_dataset_files_for_agent
from utils.fallbacks import save_splits_to_fallback, load_fallbacks_to_rundir
from utils.config import Config
from utils.exceptions import IterationRunFailed, FeedbackAgentFailed, AgentScriptFailed
from utils.snapshots import is_new_best, snapshot, get_new_and_best_metrics, replace_snapshot_path_with_relative, lock_split_files
from utils.workspace_setup import ensure_workspace_folders
from agents.architecture import run_iteration
from utils.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.report_logger import add_metrics_to_report, add_summary_to_report, rename_best_iteration_report
from utils.providers.provider import Provider, get_provider_from_string
from feedback.feedback_agent import get_feedback, get_iteration_summary
from tools.setup_tools import create_tools
from utils.snapshots import reset_snapshot_if_val_split_changed, create_split_fingerprint
from agents.steps.data_split import DataSplit

async def main(model_name, feedback_model_name, dataset, tags, val_metric, 
               workspace_dir, prepared_datasets_dir, prepared_test_sets_dir, agent_datasets_dir, iterations, 
               user_prompt, provider_name, on_new_best_callbacks, split_allowed_iterations):
    agent_id = os.getenv('AGENT_ID')
    # Initialize configuration 
    config = Config(
        agent_id=agent_id,
        model_name=model_name, 
        feedback_model_name=feedback_model_name, 
        dataset=dataset, 
        tags=tags, 
        val_metric=val_metric,
        workspace_dir=Path(workspace_dir),
        prepared_datasets_dir=Path(prepared_datasets_dir),
        prepared_test_sets_dir=Path(prepared_test_sets_dir),
        agent_datasets_dir=Path(agent_datasets_dir),
        iterations=iterations,
        user_prompt=user_prompt,
        split_allowed_iterations=split_allowed_iterations,
    )
    ensure_workspace_folders(config)
    create_run_and_snapshot_dirs(config)
    config.print_summary()
    
    # initialize logging
    if are_wandb_vars_available():
        wandb_logged_in = setup_logging(config)
    else:
        wandb_logged_in = False
    # initialize LLMs
    provider = get_provider_from_string(provider_name)
    default_model = provider.create_model(config.model_name, config)
    feedback_model = provider.create_model(config.feedback_model_name, config)
    #TODO Instantiate report logger model and pass it to add_summary_to_report

    await run_agentomics(config=config, default_model=default_model, feedback_model=feedback_model, on_new_best_callbacks=on_new_best_callbacks)

    if(wandb_logged_in):
        wandb.finish()

async def run_agentomics(config: Config, default_model, feedback_model, on_new_best_callbacks):
    tools = create_tools(config)
    
    iter_to_summary = {}
    iter_to_metrics = {}
    iter_to_feedback = {}
    last_successful_iter = None
    last_split_strategy = None
    print(f"Starting training loop with {config.iterations} iterations")
    for run_index in range(config.iterations):
        print(f"\n=== ITERATION {run_index + 1} / {config.iterations} ===")
        if(not config.can_iteration_split_data(run_index)):
            lock_split_files(config)
        split_fingerprint_before_iteration = create_split_fingerprint(config)
        try:
            # Not using feedback from failed iterations
            feedback = iter_to_feedback[last_successful_iter] if (last_successful_iter is not None) else "No feedback available"
            last_iter_summary = iter_to_summary[last_successful_iter] if (last_successful_iter is not None) else "No summary available"
            structured_outputs = await run_iteration(
                config=config,
                model=default_model, 
                iteration=run_index, 
                summary=last_iter_summary,
                feedback=feedback, 
                tools=tools,
                last_split_strategy=last_split_strategy,
            )
            last_split_strategy = next((step.splitting_strategy for step in structured_outputs if isinstance(step, DataSplit)), None)
            save_splits_to_fallback(config)
            last_successful_iter = run_index
        except IterationRunFailed as e:
            log_serial_metrics(prefix='validation', metrics=None, iteration=run_index, task_type=config.task_type)
            log_serial_metrics(prefix='train', metrics=None, iteration=run_index, task_type=config.task_type)
            #TODO also files in run dir should revert?
            load_fallbacks_to_rundir(config, run_index)
            val_split_changed = reset_snapshot_if_val_split_changed(
                config,
                iteration=run_index, 
                old_fingerprint=split_fingerprint_before_iteration, 
                new_fingerprint=create_split_fingerprint(config),
            )
            assert not val_split_changed #TODO delete
            new_metrics, best_metrics = get_new_and_best_metrics(config)
            iter_to_metrics[run_index] = new_metrics
            iter_to_feedback[run_index] = "Iteration failed, no feedback available."
            iter_to_summary[run_index] = "Iteration failed, no summary available."
            log_files(config, iteration=run_index)
            continue

        val_split_changed = reset_snapshot_if_val_split_changed(
            config,
            iteration=run_index, 
            old_fingerprint=split_fingerprint_before_iteration, 
            new_fingerprint=create_split_fingerprint(config),
        )

        extra_info = ""
        print("Starting evaluation phase")
        try:
            print("  Running validation inference...")
            run_inference_and_log(config, iteration=run_index, evaluation_stage='validation')
        except AgentScriptFailed:
            extra_info += f"Inference on validation data failed. Traceback:{traceback.format_exc()}"
        try:
            print("  Running training inference...")
            run_inference_and_log(config, iteration=run_index, evaluation_stage='train')
        except AgentScriptFailed:
            extra_info += f"Inference on train data failed. Traceback:{traceback.format_exc()}"

        new_metrics, best_metrics = get_new_and_best_metrics(config)
        iter_to_metrics[run_index] = new_metrics
        try:
            iter_to_summary[run_index] = await get_iteration_summary(
                structured_outputs=structured_outputs,
                model=feedback_model,
                config=config,
            )
            iter_to_feedback[run_index] = await get_feedback(
                structured_outputs=structured_outputs,
                config=config, 
                new_metrics=new_metrics, 
                best_metrics=best_metrics, 
                is_new_best=is_new_best(config), 
                model=feedback_model,
                iteration=run_index,
                extra_info=extra_info,
                iter_to_summary=iter_to_summary,
                iter_to_metrics=iter_to_metrics,
                val_split_changed=val_split_changed,
            )
        except FeedbackAgentFailed as e:
            iter_to_summary[run_index] = "No summary available."
            iter_to_feedback[run_index] = f"This was the {'not' if not is_new_best(config) else ''} run with best validation metrics so far. No feedback available."
            log_feedback_failure(e.exception_trace, iteration=run_index)

        if(is_new_best(config)):
            snapshot(config, run_index)  # Snapshotting overrides the previous snapshot, influencing the get_new_and_best_metrics function
            for callback in on_new_best_callbacks:
                callback(config)

        add_metrics_to_report(config, run_index)
        await add_summary_to_report(default_model, config, run_index)
        log_files(config, iteration=run_index)

    rename_best_iteration_report(config)
    log_files(config)
    export_config_to_workspace(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Runs agent and outputs results to the workspace directory")
    parser.add_argument('--dataset-name', required=True, help='Name of the folder containing dataset files')
    parser.add_argument('--model', help='LLM model to use', required=True)
    parser.add_argument('--provider', required=True, help=f'API provider to use. Available: {Provider.get_available_providers()}.')
    parser.add_argument('--workspace-dir', type=Path, default=Path('../workspace').resolve(), help='Path to a directory which will store agent runs, snapshots, and reports')
    parser.add_argument('--prepared-datasets-dir', type=Path, default=Path('../repository/prepared_datasets').resolve(), help='Path to a directory which contains prepared datasets.')
    parser.add_argument('--prepared-test-sets-dir', type=Path, default=Path('../repository/prepared_test_sets').resolve(), help='Path to a directory which contains prepared test sets.')
    parser.add_argument('--agent-datasets-dir', type=Path, default=Path('../workspace/datasets').resolve(), help='Path to a directory which contains non-test data accessible by agents.')
    parser.add_argument('--tags', nargs='*', default=[], help='(Optional) Tags for a wandb run logging')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations to run')
    parser.add_argument('--split-allowed-iterations', type=int, default=1, help='Number of initial iterations that allow the agent to split the data into training and validation sets')

    parser.add_argument('--user-prompt', type=str, default="Create the best possible machine learning model that will generalize to new unseen data.", help='(Optional) Text to overwrite the default user prompt')

    val_metric_choices = get_classification_metrics_names() + get_regression_metrics_names()
    parser.add_argument('--val-metric', help='Validation metric to use for the best model selection', required=True, choices=val_metric_choices)

    return parser.parse_args()

async def run_experiment(model, dataset_name, val_metric, prepared_datasets_dir, prepared_test_sets_dir, agent_datasets_dir,
                          workspace_dir, tags, iterations, user_prompt, provider, 
                          split_allowed_iterations=1, on_new_best_callbacks=[]):
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
        workspace_dir=workspace_dir, 
        prepared_datasets_dir=prepared_datasets_dir, 
        agent_datasets_dir=agent_datasets_dir,
        iterations=iterations,
        user_prompt=user_prompt,
        provider_name=provider,
        on_new_best_callbacks=on_new_best_callbacks,
        split_allowed_iterations=split_allowed_iterations,
        prepared_test_sets_dir=prepared_test_sets_dir,
    )


async def run_experiment_from_terminal():
    args = parse_args()

    await run_experiment(
        model=args.model, 
        dataset_name=args.dataset_name, 
        val_metric=args.val_metric, 
        prepared_datasets_dir=args.prepared_datasets_dir, 
        prepared_test_sets_dir=args.prepared_test_sets_dir,
        agent_datasets_dir=args.agent_datasets_dir, 
        workspace_dir=args.workspace_dir, 
        tags=args.tags,
        iterations=args.iterations,
        user_prompt=args.user_prompt,
        provider=args.provider,
        split_allowed_iterations=args.split_allowed_iterations,
    )

if __name__ == "__main__":
    asyncio.run(run_experiment_from_terminal())