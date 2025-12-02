import asyncio
import traceback
import argparse
from pathlib import Path
import os
import sys
import time
import json
import warnings

# Suppress noisy Pydantic warnings from dependencies
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic._internal._generate_schema')

import wandb
import weave
from timeout_function_decorator import timeout as timeout_decorator
from rich.console import Console
from rich.panel import Panel

console = Console()

from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_serial_metrics, log_feedback_failure, log_iteration_duration, log_new_best
from run_logging.wandb_setup import setup_logging
from run_logging.log_files import log_files, export_config_to_snapshot
from utils.env_utils import are_wandb_vars_available
from utils.create_user import create_run_and_snapshot_dirs
from utils.dataset_utils import setup_nonsensitive_dataset_files_for_agent
from utils.fallbacks import save_splits_to_fallback, load_fallbacks_to_rundir
from utils.config import Config
from utils.exceptions import IterationRunFailed, FeedbackAgentFailed, AgentScriptFailed
from utils.snapshots import is_new_best, snapshot, get_new_and_best_metrics, populate_iteration_dir, lock_split_files
from utils.workspace_setup import ensure_workspace_folders
from agents.architecture import run_iteration
from utils.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.report_logger import add_metrics_to_report, add_summary_to_report
from utils.providers.provider import Provider, get_provider_from_string
from feedback.feedback_agent import get_feedback
from tools.setup_tools import create_tools, get_tool_names
from utils.snapshots import reset_snapshot_if_val_split_changed, create_split_fingerprint, wipe_current_iter_files, delete_metrics_from_iteration_dir, get_best_iteration
from agents.steps.data_split import DataSplit
from utils.step_snapshots import get_latest_iteration, Step

def load_source_run_config(workspace_dir: Path, fork_from_run: str):
    """Load configuration from a source run for forking"""
    storage_config_path = Path(workspace_dir) / ".agentomics_storage" / "config.json"
    
    if not storage_config_path.exists():
        console.print(Panel(
            f"[red]Cannot fork from run '{fork_from_run}'[/red]\n\n"
            f"[bold]Reason:[/bold] Config file not found at:\n"
            f"  [dim]{storage_config_path}[/dim]\n\n"
            f"[bold]This usually means:[/bold]\n"
            f"  • The run doesn't exist yet\n"
            f"  • The run was created before config tracking was added\n\n"
            f"[bold]To fix:[/bold]\n"
            f"  1. Verify the run ID is correct\n"
            f"  2. Re-run the original experiment to generate the config\n"
            f"  3. Then try forking again",
            title="Fork Error",
            border_style="red"
        ))
        raise ValueError(f"Source run config not found: {storage_config_path}")
    
    with open(storage_config_path) as f:
        return json.load(f)

async def main(model_name, feedback_model_name, dataset, tags, val_metric,
               workspace_dir, prepared_datasets_dir, prepared_test_sets_dir, agent_datasets_dir, iterations, 
               user_prompt, provider_name, on_new_best_callbacks, split_allowed_iterations, time_deadline,
               fork_from_run=None, fork_from_step=None, fork_from_iteration=None):
    agent_id = os.getenv('AGENT_ID')
    
    # If forking, validate configuration (already loaded in run_experiment)
    source_config_dict = None
    if fork_from_run and fork_from_step:
        source_config_dict = load_source_run_config(workspace_dir, fork_from_run)
        
        # Validate dataset matches source
        source_dataset = source_config_dict.get('dataset')
        if dataset != source_dataset:
            raise ValueError(
                f"Dataset mismatch!\n"
                f"  Source run used: {source_dataset}\n"
                f"  Current config:  {dataset}\n"
                f"  This should not happen - please report as a bug."
            )
        
        # Determine what was inherited vs provided
        default_prompt = "Create the best possible machine learning model that will generalize to new unseen data."
        source_prompt = source_config_dict.get('user_prompt', default_prompt)
        source_metric = source_config_dict.get('val_metric')
        
        prompt_inherited = (user_prompt == source_prompt)
        metric_inherited = (val_metric == source_metric)
        
        # Print summary of fork configuration
        config_text = f"[bold]Dataset:[/bold]         {dataset}\n"
        config_text += f"                   [dim](from source run)[/dim]\n\n"
        config_text += f"[bold]Val Metric:[/bold]      {val_metric}\n"
        config_text += f"                   [dim]{'(from source run)' if metric_inherited else '(custom override)'}[/dim]\n\n"
        
        # Always show the actual prompt being used
        prompt_display = user_prompt if len(user_prompt) <= 60 else user_prompt[:60] + '...'
        config_text += f"[bold]User Prompt:[/bold]     [dim]{'(from source run)' if prompt_inherited else '(custom override)'}[/dim]\n"
        config_text += f"                   [italic]{prompt_display}[/italic]"
        
        console.print(Panel(config_text, title="Fork Configuration", border_style="cyan"))
    
    # Initialize configuration
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
        time_deadline=time_deadline,
    )
    ensure_workspace_folders(config)
    create_run_and_snapshot_dirs(config)
    config.print_summary()
    
    # Save config to .agentomics_storage at start of run (for forking later)
    export_config_to_snapshot(config)
    
    # Handle fork parameters
    resume_from = None
    if fork_from_run and fork_from_step:
        # Use source_config_dict loaded earlier
        if source_config_dict is None:
            source_config_dict = load_source_run_config(workspace_dir, fork_from_run)
        
        # Auto-select iteration if not specified (defaults to latest)
        if fork_from_iteration is None:
            fork_from_iteration = get_latest_iteration(config.workspace_dir, fork_from_run)
            print(f"Auto-selected latest iteration: {fork_from_iteration}")
        
        # Include source user prompt in resume_from for lineage tracking
        source_user_prompt_for_lineage = source_config_dict.get('user_prompt', user_prompt)
        resume_from = (fork_from_run, fork_from_step, fork_from_iteration, source_user_prompt_for_lineage)
        print(f"Forking from run '{fork_from_run}' at step '{fork_from_step}' (iteration {fork_from_iteration})")
    
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

    await run_agentomics(config=config, default_model=default_model, feedback_model=feedback_model, 
                        on_new_best_callbacks=on_new_best_callbacks, provider=provider, resume_from=resume_from)

    if(wandb_logged_in):
        wandb.finish()

@weave.op(call_display_name=lambda call: f"Agentomics run - agent_id: {call.inputs['config'].agent_id}")
async def run_agentomics(config: Config, default_model, feedback_model, on_new_best_callbacks, provider, resume_from=None):
    tools = create_tools(config)
    
    iter_to_outputs = {}
    iter_to_metrics = {}
    iter_to_feedback = {}
    iter_to_duration = {}
    iter_to_split_changed = {}
    last_successful_iter = None
    last_split_strategy = None
    print(f"Starting training loop with {config.iterations} iterations")
    for run_index in range(config.iterations):
        print(f"\n=== ITERATION {run_index} / {config.iterations - 1} ===")
        if(not config.can_iteration_split_data(run_index)):
            lock_split_files(config)
        split_fingerprint_before_iteration = create_split_fingerprint(config)
        start = time.time()
        try:
            # Not using feedback from failed iterations
            feedback = iter_to_feedback[last_successful_iter] if (last_successful_iter is not None) else "No instructions available"
            
            # Only use resume_from on the first iteration
            iteration_resume_from = resume_from if run_index == 0 else None
            
            structured_outputs = await run_iteration(
                config=config,
                model=default_model, 
                iteration=run_index, 
                feedback=feedback, 
                tools=tools,
                last_split_strategy=last_split_strategy,
                resume_from=iteration_resume_from,
            )
            iter_to_duration[run_index] = time.time() - start
            log_iteration_duration(iteration=run_index, duration=iter_to_duration[run_index])
            last_split_strategy = next((step.splitting_strategy for step in structured_outputs if isinstance(step, DataSplit)), None)
            save_splits_to_fallback(config)
            last_successful_iter = run_index
        except IterationRunFailed as e:
            iter_to_duration[run_index] = time.time() - start
            log_iteration_duration(iteration=run_index, duration=iter_to_duration[run_index])
            log_serial_metrics(prefix='validation', metrics=None, iteration=run_index, task_type=config.task_type)
            log_serial_metrics(prefix='train', metrics=None, iteration=run_index, task_type=config.task_type)
            load_fallbacks_to_rundir(config, run_index)
            val_split_changed = reset_snapshot_if_val_split_changed(
                config,
                iteration=run_index, 
                old_fingerprint=split_fingerprint_before_iteration, 
                new_fingerprint=create_split_fingerprint(config),
            )
            iter_to_metrics[run_index] = {}
            iter_to_feedback[run_index] = "Iteration failed, no instructions available."
            iter_to_outputs[run_index] = "Iteration failed, no outputs available."
            log_files(config, iteration=run_index)
            iter_to_split_changed[run_index] = val_split_changed
            wipe_current_iter_files(config)
            continue

        val_split_changed = reset_snapshot_if_val_split_changed(
            config,
            iteration=run_index, 
            old_fingerprint=split_fingerprint_before_iteration, 
            new_fingerprint=create_split_fingerprint(config),
        )
        iter_to_split_changed[run_index] = val_split_changed

        extra_info = ""
        print("Starting evaluation phase")
        try:
            print("  Running validation inference...")
            run_inference_and_log(config, iteration=run_index, evaluation_stage='validation')
        except AgentScriptFailed:
            exception_trace = traceback.format_exc()
            print("Validated Inference faied:\n",exception_trace)
            extra_info += f"Inference on validation data failed. Traceback:{exception_trace}"
        try:
            print("  Running training inference...")
            run_inference_and_log(config, iteration=run_index, evaluation_stage='train')
        except AgentScriptFailed:
            exception_trace = traceback.format_exc()
            print("Validated Inference faied:\n",exception_trace)
            extra_info += f"Inference on train data failed. Traceback:{exception_trace}"

        new_metrics, best_metrics = get_new_and_best_metrics(config)
        iter_to_metrics[run_index] = new_metrics
        iter_to_outputs[run_index] = structured_outputs

        is_current_new_best = is_new_best(config)
        if(is_current_new_best):
            log_new_best(iteration=run_index)
            # Snapshotting overrides the previous snapshot, influencing the get_new_and_best_metrics function
            snapshot(config, run_index, structured_outputs)
            # Also persist the latest config to .agentomics_storage for future forking
            export_config_to_snapshot(config)
            for callback in on_new_best_callbacks:
                callback(config)
        populate_iteration_dir(config, run_index, is_best=is_current_new_best, structured_outputs=structured_outputs)
        delete_metrics_from_iteration_dir(config, run_index) #needs to be there for snapshotting, but removed after to not mixup metrics from diff splits if agent runs cat on metrics

        try:
            iter_to_feedback[run_index] = await get_feedback(
                config=config, 
                is_new_best=is_current_new_best, 
                model=feedback_model,
                iteration=run_index,
                extra_info=extra_info,
                iter_to_outputs=iter_to_outputs,
                iter_to_metrics=iter_to_metrics,
                iter_to_split_changed=iter_to_split_changed,
                val_split_changed=val_split_changed,
                iter_to_duration=iter_to_duration,
                provider=provider,
                tool_names=get_tool_names(tools),
            )
        except FeedbackAgentFailed as e:
            iter_to_outputs[run_index] = "No outputs available."
            iter_to_feedback[run_index] = f"No instructions available."
            log_feedback_failure(e.exception_trace, iteration=run_index)

        add_metrics_to_report(config, run_index, new_metrics)
        await add_summary_to_report(default_model, config, run_index)
        log_files(config, iteration=run_index)
    
    # After all iterations, ensure we have a final snapshot
    # This is important for forked runs that may not produce a "new best"
    snapshot_dir = config.snapshots_dir / config.agent_id
    if not snapshot_dir.exists() or not list(snapshot_dir.glob("*.py")):
        # No snapshot exists yet, create one from the last successful iteration
        if last_successful_iter is not None:
            print(f"\nCreating final snapshot from iteration {last_successful_iter}...")
            snapshot(config, last_successful_iter, iter_to_outputs[last_successful_iter])
        else:
            print("\nWarning: No successful iterations to snapshot")


def parse_args():
    parser = argparse.ArgumentParser(description="Runs agent and outputs results to the workspace directory")
    parser.add_argument('--dataset-name', help='Name of the folder containing dataset files (auto-detected when forking)')
    parser.add_argument('--model', help='LLM model to use', required=True)
    parser.add_argument('--provider', required=True, help=f'API provider to use. Available: {Provider.get_available_providers()}.')
    parser.add_argument('--workspace-dir', type=Path, default=Path('../workspace').resolve(), help='Path to a directory which will store agent runs, snapshots, and reports')
    parser.add_argument('--prepared-datasets-dir', type=Path, default=Path('../repository/prepared_datasets').resolve(), help='Path to a directory which contains prepared datasets.')
    parser.add_argument('--prepared-test-sets-dir', type=Path, default=Path('../repository/prepared_test_sets').resolve(), help='Path to a directory which contains prepared test sets.')
    parser.add_argument('--agent-datasets-dir', type=Path, default=Path('../workspace/datasets').resolve(), help='Path to a directory which contains non-test data accessible by agents.')
    parser.add_argument('--tags', nargs='*', default=[], help='(Optional) Tags for a wandb run logging')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations to run')
    parser.add_argument("--timeout", type=int, help="Timeout before the run is shut down in seconds")
    parser.add_argument('--split-allowed-iterations', type=int, default=1, help='Number of initial iterations that allow the agent to split the data into training and validation sets')
    parser.add_argument('--user-prompt', type=str, default="Create the best possible machine learning model that will generalize to new unseen data.", help='(Optional) Custom instructions for the agent. When forking, inherits from source run unless explicitly provided.')

    val_metric_choices = get_classification_metrics_names() + get_regression_metrics_names()
    parser.add_argument('--val-metric', help='Validation metric to use for the best model selection (auto-detected when forking)', choices=val_metric_choices)

    # Fork arguments
    fork_group = parser.add_argument_group('forking', 'Fork from a previous run')
    fork_group.add_argument('--fork-from-run', type=str, help='Run ID to fork from')
    fork_group.add_argument('--fork-from-step', type=str, 
                           choices=Step.get_step_names(),
                           help='Step name to fork from')
    fork_group.add_argument('--fork-from-iteration', type=int, default=None, 
                           help='Iteration to fork from. If omitted, uses the latest (most recent) iteration.')

    return parser.parse_args()

async def run_experiment(model, dataset_name, val_metric, prepared_datasets_dir, prepared_test_sets_dir, agent_datasets_dir,
                          workspace_dir, tags, iterations, user_prompt, provider, timeout, 
                          split_allowed_iterations=1, on_new_best_callbacks=[],
                          fork_from_run=None, fork_from_step=None, fork_from_iteration=None):
    
    # If forking, load dataset/val_metric/user_prompt from source BEFORE setup
    if fork_from_run and fork_from_step:
        print(f"\n{'='*70}")
        print(f"FORK: Loading configuration from source run...")
        print(f"{'='*70}")
        
        source_config = load_source_run_config(workspace_dir, fork_from_run)
        
        # Auto-fill dataset if not provided
        if dataset_name is None:
            dataset_name = source_config.get('dataset')
            print(f"[FORK] Auto-detected dataset: {dataset_name}")
        
        # Auto-fill val_metric if not provided
        if val_metric is None:
            val_metric = source_config.get('val_metric')
            print(f"[FORK] Auto-detected validation metric: {val_metric}")
        
        # Auto-fill user_prompt if not provided
        default_prompt = "Create the best possible machine learning model that will generalize to new unseen data."
        if user_prompt == default_prompt:
            source_prompt = source_config.get('user_prompt', default_prompt)
            user_prompt = source_prompt
            print(f"[FORK] Using user prompt from source run")
        else:
            print(f"[FORK] Using custom user prompt (not from source)")
        
        print(f"[FORK] Active user_prompt: {user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}")
        print(f"{'='*70}\n")
    
    setup_nonsensitive_dataset_files_for_agent(
        prepared_datasets_dir=Path(prepared_datasets_dir),
        agent_datasets_dir=Path(agent_datasets_dir),
        dataset_name=dataset_name,
    )
    FEEDBACK_MODEL = model
    timeouted_main = timeout_decorator(timeout)(main)
    time_deadline = time.time() + timeout if timeout is not None else None
    try:
        print(f'Starting a run with a {timeout} second timeout')
        await timeouted_main(
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
            time_deadline=time_deadline,
            fork_from_run=fork_from_run,
            fork_from_step=fork_from_step,
            fork_from_iteration=fork_from_iteration,
        )
    except TimeoutError:
        print('Timeout reached')
        exit(0)


async def run_experiment_from_terminal():
    args = parse_args()
    
    # Validate required arguments (can be auto-filled when forking)
    is_forking = args.fork_from_run and args.fork_from_step
    
    if not is_forking:
        # When NOT forking, dataset and val_metric are required
        if not args.dataset_name:
            print("Error: --dataset-name is required when not forking")
            print("Use --fork-from-run to auto-detect dataset from source run")
            sys.exit(1)
        if not args.val_metric:
            print("Error: --val-metric is required when not forking")
            print("Use --fork-from-run to auto-detect metric from source run")
            sys.exit(1)

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
        timeout=args.timeout,
        fork_from_run=args.fork_from_run,
        fork_from_step=args.fork_from_step,
        fork_from_iteration=args.fork_from_iteration,
    )

if __name__ == "__main__":
    asyncio.run(run_experiment_from_terminal())