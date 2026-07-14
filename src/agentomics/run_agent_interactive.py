from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console

from agentomics.datasets.data_contract import VALIDATION_SPLIT
from agentomics.datasets.dataset_preparation import prepare_dataset
from agentomics.datasets.datasets_interactive import (
    get_all_datasets_info,
    interactive_dataset_selection,
    print_datasets_table,
)
from agentomics.run_agent import run_experiment
from agentomics.run_logging.env_utils import are_wandb_vars_available
from agentomics.runtime.read_write_utils import get_next_iteration_index, load_config_from_run_dir
from agentomics.utils.config import Config
from agentomics.utils.metrics import resolve_val_metric
from agentomics.utils.metrics_interactive import display_metrics_table
from agentomics.utils.printing_utils import print_phase
from agentomics.utils.providers.provider import Provider, get_provider_and_api_key
from agentomics.utils.user_input import get_user_input_for_int


console = Console()


def resolve_run_arguments(arguments: argparse.Namespace) -> None:
    existing_config = load_config_from_run_dir(
        arguments.workspace_dir / Config.RUN_DIRNAME,
        missing_ok=True,
    )
    if existing_config is not None:
        next_iteration_index = get_next_iteration_index(existing_config)
        if arguments.dataset is not None and arguments.dataset != existing_config.dataset:
            console.print(
                f"Warning: --dataset is ignored for forked runs. "
                f"Using '{existing_config.dataset}' from the source run config.",
                style="red",
            )
        if arguments.val_metric is not None and arguments.val_metric != existing_config.val_metric:
            console.print(
                f"Warning: --val-metric is ignored for forked runs. "
                f"Using '{existing_config.val_metric}' from the source run config.",
                style="red",
            )
        arguments.dataset = existing_config.dataset
        arguments.val_metric = existing_config.val_metric
        arguments.model = arguments.model or existing_config.model_name
        arguments.iteration_plan_model = (
            arguments.iteration_plan_model or existing_config.iteration_plan_model_name
        )
        arguments.provider = arguments.provider or existing_config.provider_name
        arguments.iterations = (
            existing_config.iterations
            if arguments.iterations is None
            else next_iteration_index + arguments.iterations
        )
        arguments.tags = arguments.tags if arguments.tags is not None else existing_config.tags
        arguments.split_allowed_iterations = (
            existing_config.split_allowed_iterations
            if arguments.split_allowed_iterations is None
            else next_iteration_index + arguments.split_allowed_iterations
        )
        arguments.exploration_iterations = (
            existing_config.exploration_iterations
            if arguments.exploration_iterations is None
            else next_iteration_index + arguments.exploration_iterations
        )
        arguments.run_python_timeout = (
            existing_config.run_python_tool_timeout
            if arguments.run_python_timeout is None
            else arguments.run_python_timeout
        )
        arguments.user_prompt = arguments.user_prompt or existing_config.user_prompt

    if arguments.tags is None:
        arguments.tags = []
    if arguments.split_allowed_iterations is None:
        arguments.split_allowed_iterations = Config.DEFAULT_SPLIT_ALLOWED_ITERATIONS
    if arguments.exploration_iterations is None:
        arguments.exploration_iterations = Config.DEFAULT_EXPLORATION_ITERATIONS
    if arguments.run_python_timeout is None:
        arguments.run_python_timeout = Config.DEFAULT_RUN_PYTHON_TOOL_TIMEOUT
    if arguments.user_prompt is None:
        arguments.user_prompt = Config.DEFAULT_USER_PROMPT


def _is_tty_available() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _resolve_interactive_parameters(
    arguments: argparse.Namespace,
    provider: Provider,
) -> tuple[str, str, int]:
    if (not arguments.dataset or not arguments.model) and not _is_tty_available():
        raise RuntimeError(
            "Non-interactive runs require --dataset and --model, or --fork-from-run"
        )

    dataset = arguments.dataset
    if not dataset:
        print_phase("Dataset Selection")
        dataset = interactive_dataset_selection(
            get_all_datasets_info(arguments.datasets_dir)
        )
        if not dataset:
            raise RuntimeError("No dataset selected")

    model = arguments.model
    if not model:
        print_phase("Model Selection")
        model = provider.interactive_model_selection(limit=50)

    iterations = arguments.iterations
    if iterations is None:
        print_phase("Run Configuration")
        iterations = (
            get_user_input_for_int(
                "Enter number of iterations to run (Recommended more than 5):",
                default=Config.DEFAULT_ITERATIONS,
            )
            if _is_tty_available()
            else Config.DEFAULT_ITERATIONS
        )
    return dataset, model, iterations


def run_agent_interactive(arguments: argparse.Namespace) -> int:
    resolve_run_arguments(arguments)

    if arguments.list_datasets:
        console.print("Available Datasets", style="cyan")
        print_datasets_table(get_all_datasets_info(arguments.datasets_dir))
        return 0
    if arguments.list_metrics:
        console.print("Available Validation Metrics", style="cyan")
        display_metrics_table()
        return 0

    api_key, provider_name = get_provider_and_api_key(
        preferred_provider=arguments.provider
    )
    provider = Provider.create_provider(provider_name, api_key)
    if arguments.list_models:
        console.print("Available Large Language Models", style="cyan")
        provider.display_models()
        return 0

    if not are_wandb_vars_available():
        console.print(
            "Wandb environment variables are not set. Logging is disabled.",
            style="yellow",
        )

    dataset, model, iterations = _resolve_interactive_parameters(arguments, provider)
    iteration_plan_model = arguments.iteration_plan_model or model
    prepared_datasets_dir = arguments.workspace_dir / "prepared_datasets"
    dataset_metadata = prepare_dataset(
        source_dir=arguments.datasets_dir / dataset,
        destination_dir=prepared_datasets_dir / dataset,
        task_type=arguments.task_type,
        interactive=_is_tty_available(),
    )
    task_type = dataset_metadata["task_type"]
    val_metric = resolve_val_metric(task_type, arguments.val_metric)
    split_allowed_iterations = arguments.split_allowed_iterations
    if (prepared_datasets_dir / dataset / VALIDATION_SPLIT).exists():
        split_allowed_iterations = 0

    asyncio.run(
        run_experiment(
            model=model,
            iteration_plan_model=iteration_plan_model,
            dataset_name=dataset,
            task_type=task_type,
            label_to_scalar=dataset_metadata.get("label_to_scalar"),
            input_structure=dataset_metadata["input_structure"],
            val_metric=val_metric,
            datasets_dir=prepared_datasets_dir,
            workspace_dir=arguments.workspace_dir,
            tags=arguments.tags,
            iterations=iterations,
            user_prompt=arguments.user_prompt,
            provider=provider_name,
            split_allowed_iterations=split_allowed_iterations,
            exploration_iterations=arguments.exploration_iterations,
            timeout=arguments.timeout,
            split_timeout=arguments.split_timeout,
            run_python_timeout=arguments.run_python_timeout,
            disable_training_reporting=arguments.disable_training_reporting,
            conda_export_mode=arguments.conda_export_mode,
        )
    )
    return 0
