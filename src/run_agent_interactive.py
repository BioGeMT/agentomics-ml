import argparse
import asyncio
import sys
from pathlib import Path

import dotenv
from rich.console import Console

from run_agent import run_experiment
from runtime.read_write_utils import get_next_iteration_index, load_config_from_run_dir
from utils.config import Config
from datasets.dataset_utils import get_all_prepared_datasets_info, get_task_type_from_prepared_dataset
from datasets.datasets_interactive import interactive_dataset_selection, print_datasets_table
from run_logging.env_utils import are_wandb_vars_available
from utils.metrics import get_classification_metrics_names, get_regression_metrics_names, resolve_val_metric
from utils.metrics_interactive import display_metrics_table
from utils.providers.provider import Provider, get_provider_and_api_key
from utils.printing_utils import print_phase
from utils.user_input import get_user_input_for_int

console = Console()


# ── CLI argument parsing ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Agentomics-ML Entry Point")

    # List-only modes
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")
    parser.add_argument("--list-metrics", action="store_true", help="List available validation metrics and exit")

    # Run configuration
    parser.add_argument("--dataset", help="Dataset name. If not provided, can be interactively selected. Cannot be changed for forked runs — always inherited from the source run config.")
    parser.add_argument("--model", help="Model name compatible with the selected provider. If not provided, can be interactively selected. For forked runs, inherits from the source run config.")
    parser.add_argument("--iteration-plan-model", help="Iteration-plan model name compatible with the selected provider. If not provided, it defaults to --model. For forked runs, inherits from the source run config.")
    parser.add_argument("--provider", help="API provider. If not provided, auto-detected from environment. For forked runs, inherits from the source run config.")
    parser.add_argument(
        "--val-metric",
        help="Validation metric. If not provided, defaults to AUROC for classification or MAE for regression. Cannot be changed for forked runs — always inherited from the source run config.",
        choices=get_classification_metrics_names() + get_regression_metrics_names(),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run. If not provided, can be interactively selected. For forked runs, omitting it keeps the source run's total iteration limit; providing it means that many additional iterations from the fork point.",
    )
    parser.add_argument(
        "--split-allowed-iterations",
        type=int,
        default=None,
        help=f"Number of initial iterations that allow the agent to split the data into training and validation sets. If not provided, defaults to {Config.DEFAULT_SPLIT_ALLOWED_ITERATIONS}. For forked runs, omitting it keeps the source run's split-allowed limit; providing it means that many more split-allowed iterations from the fork point.",
    )
    parser.add_argument(
        "--exploration-iterations",
        type=int,
        default=None,
        help=f"Number of initial iterations that should focus on baseline/exploration models. If not provided, defaults to {Config.DEFAULT_EXPLORATION_ITERATIONS}. For forked runs, omitting it keeps the source run's exploration limit; providing it means that many more exploration iterations from the fork point.",
    )
    parser.add_argument("--timeout", type=int, help="Timeout before the run is shut down in seconds")
    parser.add_argument(
        "--split-timeout",
        type=int,
        help="Timeout before the data splitting is no longer allowed in seconds. If not provided, split iterations are used as the limit.",
    )
    parser.add_argument(
        "--run-python-timeout",
        type=int,
        default=None,
        help=f"Timeout in seconds for each run_python tool execution. If not provided, defaults to {Config.DEFAULT_RUN_PYTHON_TOOL_TIMEOUT}. For forked runs, inherits from the source run config.",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help="Main goal for the agent. If not provided, defaults to the built-in default prompt. For forked runs, inherits from the source run config.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Tags to associate with the run. If not provided, defaults to no tags. For forked runs, inherits from the source run config.",
    )
    parser.add_argument(
        "--foundation-models-type",
        type=str,
        default=None,
        help="Foundation model type to enable (dna, rna, molecule, protein, all). If not provided, no foundation models are used. For forked runs, inherits from the source run config.",
    )
    parser.add_argument(
        "--foundation-models-yaml",
        type=str,
        default=None,
        help="Path to the foundation models YAML config file. For forked runs, inherits from the source run config.",
    )

    # Paths
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        required=True,
        help="Path to a directory for agent run, best iteration snapshot, and reports.",
    )
    parser.add_argument(
        "--prepared-datasets-dir",
        type=Path,
        required=True,
        help="Path to a directory containing prepared datasets.",
    )
    parser.add_argument(
        "--disable-training-reporting",
        action="store_true",
        help="Disable the TrainingReporter helper that emits structured best-effort training updates (enabled by default).",
    )

    # Misc
    parser.add_argument(
        "--root-privileges",
        action="store_true",
        help="Whether the script has root privileges to create a new user for the agent (recommended)",
    )

    args = parser.parse_args()
    args.workspace_dir = args.workspace_dir.resolve()
    args.prepared_datasets_dir = args.prepared_datasets_dir.resolve()

    # When parsing inside a forked workspace, merge CLI overrides with the source run config.
    existing_config = load_config_from_run_dir(args.workspace_dir / Config.RUN_DIRNAME, missing_ok=True)
    if existing_config is not None:
        next_iteration_index = get_next_iteration_index(existing_config)
        if args.dataset is not None and args.dataset != existing_config.dataset:
            console.print(f"Warning: --dataset is ignored for forked runs. Using '{existing_config.dataset}' from the source run config.", style="red")
        if args.val_metric is not None and args.val_metric != existing_config.val_metric:
            console.print(f"Warning: --val-metric is ignored for forked runs. Using '{existing_config.val_metric}' from the source run config.", style="red")
        args.dataset = existing_config.dataset
        args.val_metric = existing_config.val_metric
        if args.model is None:                     args.model = existing_config.model_name
        if args.iteration_plan_model is None:      args.iteration_plan_model = existing_config.iteration_plan_model_name
        if args.provider is None:                  args.provider = existing_config.provider_name
        # For forked runs, explicit iteration-limit flags mean "N more from the fork point".
        args.iterations = existing_config.iterations if args.iterations is None else next_iteration_index + args.iterations
        if args.tags is None:                      args.tags = existing_config.tags
        args.split_allowed_iterations = (
            existing_config.split_allowed_iterations
            if args.split_allowed_iterations is None
            else next_iteration_index + args.split_allowed_iterations
        )
        args.exploration_iterations = (
            existing_config.exploration_iterations
            if args.exploration_iterations is None
            else next_iteration_index + args.exploration_iterations
        )
        if args.run_python_timeout is None:        args.run_python_timeout = existing_config.run_python_tool_timeout
        if args.user_prompt is None:               args.user_prompt = existing_config.user_prompt
        if args.foundation_models_type is None:    args.foundation_models_type = existing_config.foundation_models_type
        if args.foundation_models_yaml is None:    args.foundation_models_yaml = existing_config.foundation_models_yaml

    # Apply built-in defaults only for still-unset optional args on fresh runs.
    if args.tags is None:                      args.tags = []
    if args.split_allowed_iterations is None:  args.split_allowed_iterations = Config.DEFAULT_SPLIT_ALLOWED_ITERATIONS
    if args.exploration_iterations is None:    args.exploration_iterations = Config.DEFAULT_EXPLORATION_ITERATIONS
    if args.run_python_timeout is None:        args.run_python_timeout = Config.DEFAULT_RUN_PYTHON_TOOL_TIMEOUT
    if args.user_prompt is None:               args.user_prompt = Config.DEFAULT_USER_PROMPT

    return args


# ── List-only modes ──────────────────────────────────────────────────

def handle_list_modes(args, provider: Provider) -> int | None:
    """Handle --list-datasets, --list-models, --list-metrics. Returns exit code or None to continue."""
    if args.list_datasets:
        console.print("Available Datasets", style="cyan")
        datasets = get_all_prepared_datasets_info(args.prepared_datasets_dir)
        print_datasets_table(datasets)
        return 0

    if args.list_models:
        console.print("Available Large Language Models", style="cyan")
        provider.display_models()
        return 0

    if args.list_metrics:
        console.print("Available Validation Metrics", style="cyan")
        display_metrics_table()
        return 0

    return None


# ── Interactive prompts ──────────────────────────────────────────────

def _is_tty_available() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()

def resolve_interactive_params(args, provider: Provider) -> tuple[str, str, int]:
    """Prompt for dataset, model, and iterations if not provided via CLI. Returns (dataset, model, iterations)."""
    needs_interactive = not args.dataset or not args.model
    if needs_interactive and not _is_tty_available():
        console.print("Interactive terminal required for dataset/model selection but not available", style="red")
        console.print("For non-interactive use, specify --dataset and --model arguments", style="cyan")
        raise SystemExit(1)

    dataset = args.dataset
    if not dataset:
        print_phase("Dataset Selection")
        datasets = get_all_prepared_datasets_info(args.prepared_datasets_dir)
        dataset = interactive_dataset_selection(datasets)
        if not dataset:
            console.print("No dataset selected", style="red")
            raise SystemExit(1)

    model = args.model
    if not model:
        print_phase("Model Selection")
        model = provider.interactive_model_selection(limit=50)

    iterations = args.iterations
    if iterations is None:
        print_phase("Run Configuration")
        if _is_tty_available():
            iterations = get_user_input_for_int(
                "Enter number of iterations to run (Recommended more than 5):",
                default=Config.DEFAULT_ITERATIONS,
            )
        else:
            iterations = Config.DEFAULT_ITERATIONS

    return dataset, model, iterations


# ── Main entry point ─────────────────────────────────────────────────

def main():
    args = parse_args()
    dotenv.load_dotenv()

    api_key, provider_name = get_provider_and_api_key(preferred_provider=args.provider)
    provider = Provider.create_provider(provider_name, api_key)

    exit_code = handle_list_modes(args, provider)
    if exit_code is not None:
        return exit_code

    if not are_wandb_vars_available():
        console.print("Wandb env variables not set. Logging to WANDB is disabled.", style="yellow")
        console.print(
            "To setup wandb, provide WANDB_API_KEY, WANDB_PROJECT_NAME, and WANDB_ENTITY env variables",
            style="yellow",
        )

    dataset, model, iterations = resolve_interactive_params(args, provider)
    iteration_plan_model = args.iteration_plan_model or model

    task_type = get_task_type_from_prepared_dataset(args.prepared_datasets_dir / dataset)
    val_metric = resolve_val_metric(task_type, args.val_metric)

    split_allowed_iterations = args.split_allowed_iterations
    if (args.prepared_datasets_dir / dataset / "validation.csv").exists():
        split_allowed_iterations = 0

    asyncio.run(
        run_experiment(
            model=model,
            iteration_plan_model=iteration_plan_model,
            dataset_name=dataset,
            task_type=task_type,
            val_metric=val_metric,
            prepared_datasets_dir=args.prepared_datasets_dir,
            workspace_dir=args.workspace_dir,
            tags=args.tags,
            iterations=iterations,
            user_prompt=args.user_prompt,
            provider=provider_name,
            split_allowed_iterations=split_allowed_iterations,
            exploration_iterations=args.exploration_iterations,
            timeout=args.timeout,
            split_timeout=args.split_timeout,
            run_python_timeout=args.run_python_timeout,
            foundation_models_type=args.foundation_models_type,
            foundation_models_yaml=args.foundation_models_yaml,
            disable_training_reporting=args.disable_training_reporting,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
