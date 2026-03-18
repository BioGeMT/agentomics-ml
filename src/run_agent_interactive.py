import argparse
import sys
import asyncio
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import dotenv

from utils.dataset_utils import get_all_prepared_datasets_info
from utils.datasets_interactive_utils import interactive_dataset_selection, print_datasets_table
from utils.metrics_interactive_utils import display_metrics_table
from utils.providers.provider import Provider, get_provider_and_api_key
from utils.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.env_utils import are_wandb_vars_available
from utils.user_input import get_user_input_for_int
from run_agent import run_experiment

console = Console()

def print_welcome():
    welcome_text = """
===============================================
Welcome to Agentomics-ML
===============================================
"""    
    console.print(Panel(welcome_text, style="bold blue"))

def check_tty_available():
    """Check if TTY is available for interactive operations."""
    return sys.stdin.isatty() and sys.stdout.isatty()

def main():
    """Interactive script for Agentomics-ML"""
    parser = argparse.ArgumentParser(description="Agentomics-ML Entry Point")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")
    parser.add_argument("--list-metrics", action="store_true", help="List available validation metrics and exit")
    parser.add_argument("--root-privileges", action="store_true", help="Whether the script has root privileges to create a new user for the agent (recommended)")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--iterations", type=int, help="Number of iterations to run")
    parser.add_argument("--timeout", type=int, help="Timeout before the run is shut down in seconds")
    parser.add_argument("--run-python-timeout", type=int, default=None, help="Timeout in seconds for each run_python tool execution (default: 21600)")
    parser.add_argument("--exploration-iterations",type=int,help="Number of initial iterations that should focus on baseline/exploration models",default=4)
    parser.add_argument("--split-timeout", type=int, help="Timeout before the data splitting is no longer allowed in seconds. If not provided, split iterations are used as the limit.")
    parser.add_argument("--split-allowed-iterations", type=int, help="Number of initial iterations that are allowed to (re)split the data into train/validation", default=1)
    parser.add_argument("--tags", nargs="*", default=[], help="(Optional) Comma-separated tags to associate with the run")
    parser.add_argument('--user-prompt', type=str, default="Develop a machine learning model that generalizes well to new unseen data.", help='(Optional) Text to overwrite the default user prompt')
    parser.add_argument("--model", help="Model name. Should be compatible with the selected provider")
    parser.add_argument("--provider", help="(Optional) Preferred provider to use when multiple api-key provided.")

    available_metrics = get_classification_metrics_names() + get_regression_metrics_names()
    parser.add_argument("--val-metric", help="Validation metric", choices=available_metrics)
    
    args = parser.parse_args()
    dotenv.load_dotenv()

    dataset = args.dataset
    model = args.model
    val_metric = args.val_metric
    iterations = args.iterations

    repository_dir = Path(__file__).parent.parent.resolve()
    repository_parent_dir = repository_dir.parent.resolve()
    workspace_dir = Path(os.environ.get("AGENTOMICS_WORKSPACE_DIR", str(repository_parent_dir / "workspace"))).resolve()
    paths = {
        "datasets_dir": str(repository_dir / "datasets"),
        "prepared_datasets_dir": str(repository_dir / "prepared_datasets"),
        "prepared_test_sets_dir": str(repository_dir / "prepared_test_sets"),
        "workspace_dir": str(workspace_dir),
        "agent_datasets_dir": str(workspace_dir / "datasets")
    }

    api_key, provider_name = get_provider_and_api_key(preferred_provider=args.provider)
    provider = Provider.create_provider(provider_name, api_key)

    # Handle list-only modes (these don't require interactivity)
    if args.list_datasets:
        console.print("Available Datasets", style="cyan")
        datasets = get_all_prepared_datasets_info(paths["prepared_datasets_dir"], paths["prepared_test_sets_dir"])
        print_datasets_table(datasets)
        return 0
    
    if args.list_models:
        console.print("Available Large Language Models", style="cyan")
        provider.display_models()
        return 0
    
    if args.list_metrics:
        console.print("Available Validation Metrics", style="cyan")
        display_metrics_table()  # Show all metrics when listing
        return 0
    
    # For interactive mode (when dataset/model missing), require interactive terminal
    if (not dataset or not model) and not check_tty_available():
        console.print("Interactive terminal required for dataset/model selection but not available", style="red")
        console.print("For non-interactive use, specify --dataset and --model arguments", style="cyan")
        console.print("Example: python agentomics-entrypoint.py --dataset breast_cancer --model 'openai/gpt-4'", style="cyan")
        return 1
    
    if not are_wandb_vars_available():
        console.print("Wandb env variables not set. Logging to WANDB is disabled.", style="yellow")
        console.print("To setup wandb, provide WANDB_API_KEY, WANDB_PROJECT_NAME, and WANDB_ENTITY env variables", style="yellow")
    
    # Go to interactive selection if dataset/model not provided
    print_welcome()
    if not dataset:
        datasets = get_all_prepared_datasets_info(paths["prepared_datasets_dir"], paths["prepared_test_sets_dir"])
        dataset = interactive_dataset_selection(datasets)
        if not dataset:
            console.print("No dataset selected", style="red")
            return 1
    
    if not model:
        model = provider.interactive_model_selection(limit=50)

    if not iterations:
        iterations = get_user_input_for_int("Enter number of iterations to run (Recommended more than 5):", default=5)
    
    # Run the agent
    asyncio.run(run_experiment(
        model=model,
        dataset_name=dataset,
        val_metric=val_metric,
        prepared_datasets_dir=paths["prepared_datasets_dir"],
        prepared_test_sets_dir=paths["prepared_test_sets_dir"],
        agent_datasets_dir=paths["agent_datasets_dir"],
        workspace_dir=paths["workspace_dir"],
        tags=args.tags,
        iterations=iterations,
        user_prompt=args.user_prompt,
        provider=provider_name,
        split_allowed_iterations=args.split_allowed_iterations,
        exploration_iterations=args.exploration_iterations,
        timeout=args.timeout,
        split_timeout=args.split_timeout,
        run_python_timeout=args.run_python_timeout
    ))
    return 0
        
if __name__ == "__main__":
    sys.exit(main())
