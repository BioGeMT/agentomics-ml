import argparse
import asyncio
import sys
from pathlib import Path

import dotenv
from rich.console import Console

from run_agent import run_experiment
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
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--model", help="Model name. Should be compatible with the selected provider")
    parser.add_argument("--provider", help="API provider to use. Auto-detected from env if not provided")
    parser.add_argument(
        "--val-metric",
        help="Validation metric (defaults to AUROC for classification, MAE for regression)",
        choices=get_classification_metrics_names() + get_regression_metrics_names(),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (default: None, prompted interactively)",
    )
    parser.add_argument(
        "--split-allowed-iterations",
        type=int,
        default=Config.DEFAULT_SPLIT_ALLOWED_ITERATIONS,
        help="Number of initial iterations that allow the agent to split the data into training and validation sets (default: %(default)s)",
    )
    parser.add_argument(
        "--exploration-iterations",
        type=int,
        default=Config.DEFAULT_EXPLORATION_ITERATIONS,
        help="Number of initial iterations that should focus on baseline/exploration models (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=int, help="Timeout before the run is shut down in seconds")
    parser.add_argument(
        "--split-timeout",
        type=int,
        help="Timeout before the data splitting is no longer allowed in seconds. "
        "If not provided, split iterations are used as the limit.",
    )
    parser.add_argument(
        "--run-python-timeout",
        type=int,
        default=Config.DEFAULT_RUN_PYTHON_TOOL_TIMEOUT,
        help="Timeout in seconds for each run_python tool execution (default: %(default)s)",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=Config.DEFAULT_USER_PROMPT,
        help="Text to overwrite the default user prompt",
    )
    parser.add_argument("--tags", nargs="*", default=[], help="Tags to associate with the run")
    parser.add_argument(
        "--foundation-models-type",
        type=str,
        default=None,
        help="Foundation model type to enable (dna, rna, molecule, protein, all)",
    )
    parser.add_argument(
        "--foundation-models-yaml",
        type=str,
        default=None,
        help="Path to the foundation models YAML config file",
    )

    # Paths
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        required=True,
        help="Path to a directory for agent runs, snapshots, and reports.",
    )
    parser.add_argument(
        "--prepared-datasets-dir",
        type=Path,
        required=True,
        help="Path to a directory containing prepared datasets.",
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

    task_type = get_task_type_from_prepared_dataset(args.prepared_datasets_dir / dataset)
    val_metric = resolve_val_metric(task_type, args.val_metric)

    split_allowed_iterations = args.split_allowed_iterations
    if (args.prepared_datasets_dir / dataset / "validation.csv").exists():
        split_allowed_iterations = 0

    asyncio.run(
        run_experiment(
            model=model,
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
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
