import os
import time
from pathlib import Path

import wandb
from timeout_function_decorator import timeout as timeout_decorator

from agentomics.run_logging.wandb_setup import setup_logging
from agentomics.runtime.git_checkpoints import initialize_repo_if_needed
from agentomics.runtime.read_write_utils import (
    initialize_run_directories,
    save_config,
    save_dataset_metadata,
)
from agentomics.runtime.run_lifecycle import run_agentomics
from agentomics.utils.config import Config
from agentomics.run_logging.env_utils import are_wandb_vars_available
from agentomics.utils.printing_utils import print_phase
from agentomics.utils.providers.provider import Provider


def initialize_run(config: Config, dataset_metadata: dict) -> None:
    initialize_run_directories(config)
    config.wandb_run_id = setup_logging(config) if are_wandb_vars_available() else None
    save_config(config)
    save_dataset_metadata(config, dataset_metadata)
    initialize_repo_if_needed(config)


async def run_experiment(
    model_name: str,
    iteration_plan_model_name: str,
    dataset_name: str,
    task_type: str,
    val_metric: str,
    datasets_dir: str | Path,
    workspace_dir: str | Path,
    tags: list[str],
    iterations: int,
    user_prompt: str,
    provider: Provider,
    timeout: int | None,
    split_timeout: int | None,
    run_python_timeout: int,
    split_allowed_iterations: int,
    exploration_iterations: int,
    input_structure: list[str],
    dataset_metadata: dict,
    label_to_scalar: dict[str, int] | None = None,
    disable_training_reporting: bool = False,
    conda_export_mode: str = "full"
):
    workspace_dir = Path(workspace_dir)
    datasets_dir = Path(datasets_dir)

    time_deadline = time.time() + timeout if timeout is not None else None
    split_time_deadline = time.time() + split_timeout if split_timeout is not None else None

    agent_id = os.getenv("AGENT_ID")
    config = Config(
        agent_id=agent_id,
        model_name=model_name,
        iteration_plan_model_name=iteration_plan_model_name,
        provider_name=provider.name,
        dataset=dataset_name,
        tags=tags,
        val_metric=val_metric,
        workspace_dir=str(workspace_dir),
        datasets_dir=str(datasets_dir),
        iterations=iterations,
        user_prompt=user_prompt,
        task_type=task_type,
        label_to_scalar=label_to_scalar,
        input_structure=input_structure,
        split_allowed_iterations=split_allowed_iterations,
        exploration_iterations=exploration_iterations,
        time_deadline=time_deadline,
        split_time_deadline=split_time_deadline,
        run_python_tool_timeout=run_python_timeout,
        disable_training_reporting=disable_training_reporting,
        conda_export_mode=conda_export_mode
    )
    print_phase("Agentomics run started")
    initialize_run(config, dataset_metadata)

    config.print_summary()

    default_model = provider.create_model(config.model_name, config)
    iteration_plan_model = provider.create_model(config.iteration_plan_model_name, config)

    run_fn = run_agentomics
    if timeout is not None:
        run_fn = timeout_decorator(timeout)(run_agentomics)

    try:
        print(f"Starting a run with a {timeout} second timeout")
        await run_fn(
            config=config,
            default_model=default_model,
            iteration_plan_model=iteration_plan_model,
            provider=provider,
        )
    except TimeoutError:
        print("Timeout reached")

    if config.wandb_run_id is not None:
        wandb.finish()
