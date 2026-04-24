import os
import time
from pathlib import Path

import wandb
from timeout_function_decorator import timeout as timeout_decorator

from run_logging.wandb_setup import setup_logging
from runtime.git_checkpoints import initialize_repo_if_needed
from runtime.read_write_utils import initialize_run_directories, save_config
from runtime.run_lifecycle import run_agentomics
from utils.config import Config
from datasets.dataset_utils import setup_nonsensitive_dataset_files_for_agent
from run_logging.env_utils import are_wandb_vars_available
from utils.printing_utils import print_phase
from utils.providers.provider import get_provider_from_string


async def run_experiment(
    model: str,
    iteration_plan_model: str,
    dataset_name: str,
    task_type: str,
    val_metric: str,
    prepared_datasets_dir: str | Path,
    workspace_dir: str | Path,
    tags: list[str],
    iterations: int,
    user_prompt: str,
    provider: str,
    timeout: int | None,
    split_timeout: int | None,
    run_python_timeout: int,
    split_allowed_iterations: int,
    exploration_iterations: int,
    foundation_models_type: str | None = None,
    foundation_models_yaml: str | None = None,
):
    workspace_dir = Path(workspace_dir)
    prepared_datasets_dir = Path(prepared_datasets_dir)

    time_deadline = time.time() + timeout if timeout is not None else None
    split_time_deadline = time.time() + split_timeout if split_timeout is not None else None

    agent_id = os.getenv("AGENT_ID")
    agent_user = os.getenv("AGENT_USER")
    config = Config(
        agent_id=agent_id,
        model_name=model,
        iteration_plan_model_name=iteration_plan_model,
        provider_name=provider,
        dataset=dataset_name,
        tags=tags,
        val_metric=val_metric,
        workspace_dir=str(workspace_dir),
        prepared_datasets_dir=str(prepared_datasets_dir),
        iterations=iterations,
        user_prompt=user_prompt,
        task_type=task_type,
        split_allowed_iterations=split_allowed_iterations,
        exploration_iterations=exploration_iterations,
        time_deadline=time_deadline,
        split_time_deadline=split_time_deadline,
        run_python_tool_timeout=run_python_timeout,
        foundation_models_type=foundation_models_type,
        foundation_models_yaml=foundation_models_yaml,
        agent_user=agent_user,
    )
    print_phase("Agentomics run started")
    initialize_run_directories(config)
    setup_nonsensitive_dataset_files_for_agent(
        prepared_datasets_dir=config.prepared_dataset_dir.parent,
        agent_datasets_dir=config.agent_dataset_dir.parent,
        dataset_name=dataset_name,
    )
    config.wandb_run_id = setup_logging(config) if are_wandb_vars_available() else None
    save_config(config)
    initialize_repo_if_needed(config)

    config.print_summary()

    provider_obj = get_provider_from_string(config.provider_name)
    default_model = provider_obj.create_model(config.model_name, config)
    iteration_plan_model = provider_obj.create_model(config.iteration_plan_model_name, config)

    run_fn = run_agentomics
    if timeout is not None:
        run_fn = timeout_decorator(timeout)(run_agentomics)

    try:
        print(f"Starting a run with a {timeout} second timeout")
        await run_fn(
            config=config,
            default_model=default_model,
            iteration_plan_model=iteration_plan_model,
            provider=provider_obj,
        )
    except TimeoutError:
        print("Timeout reached")
        raise SystemExit(0)

    if config.wandb_run_id is not None:
        wandb.finish()
