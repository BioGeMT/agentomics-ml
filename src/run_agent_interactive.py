import argparse
import os
import sys
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import json
import dotenv

from utils.dataset_utils import get_all_datasets_info
from utils.datasets_interactive_utils import interactive_dataset_selection, print_datasets_table
from utils.openrouter_models import display_models, load_available_models, interactive_model_selection
from utils.metrics_interactive_utils import display_metrics_table, interactive_metric_selection
from utils.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.env_utils import is_openrouter_key_available, is_wandb_key_available
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
    parser.add_argument("--model", help="Model name")
    parser.add_argument('--user-prompt', type=str, default="Create the best possible machine learning model that will generalize to new unseen data.", help='(Optional) Text to overwrite the default user prompt')

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
    paths = {
        "datasets_dir": str(repository_dir / "datasets"),
        "prepared_datasets_dir": str(repository_dir / "prepared_datasets"),
        "workspace_dir": str(repository_parent_dir / "workspace"), 
        "agent_datasets_dir": str(repository_parent_dir / "workspace" / "datasets")
    }

    # Handle list-only modes (these don't require interactivity)
    if args.list_datasets:
        console.print("Available Datasets", style="cyan")
        datasets = get_all_datasets_info(paths["datasets_dir"], paths["prepared_datasets_dir"])
        print_datasets_table(datasets)
        return 0
    
    if args.list_models:
        #TODO only if we use openrouter - we might allow other APIs as well
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        console.print("Available Large Language Models", style="cyan")
        display_models(load_available_models(openrouter_api_key, limit=30))
        return 0
    
    if args.list_metrics:
        console.print("Available Validation Metrics", style="cyan")
        display_metrics_table()  # Show all metrics when listing
        return 0
    
    # For interactive mode (when dataset/model/val_metric missing), require interactive terminal
    if (not dataset or not model or not args.val_metric) and not check_tty_available():
        console.print("Interactive terminal required for dataset/model/val_metric selection but not available", style="red")
        console.print("For non-interactive use, specify --dataset, --model, and --val-metric arguments", style="cyan")
        console.print("Example: python agentomics-entrypoint.py --dataset heart_disease --model 'openai/gpt-4' --val-metric 'ACC'", style="cyan")
        return 1
    
    if not is_openrouter_key_available():
        console.print("OPENROUTER_API_KEY not set. Please set it to use OpenRouter models.", style="red")
        return 1
    if not is_wandb_key_available():
        console.print("WANDB_API_KEY not set. Logging to WANDB is disabled.", style="yellow")
    
    # Go to interactive selection if dataset/model/val_metric not provided
    print_welcome()
    if not dataset:
        datasets = get_all_datasets_info(paths["datasets_dir"], paths["prepared_datasets_dir"])
        dataset = interactive_dataset_selection(datasets)
        if not dataset:
            console.print("No dataset selected", style="red")
            return 1
    
    # Load metadata
    prepared_dataset_path = Path(paths["prepared_datasets_dir"]) / dataset
    metadata_path = prepared_dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    task_type = metadata.get("task_type")

    if not val_metric:
        val_metric = interactive_metric_selection(task_type)
    
    if not model:
        model = interactive_model_selection(limit=50, default="openai/gpt-4.1")

    if not iterations:
        iterations = get_user_input_for_int("Enter number of iterations to run:", default=5)
    
    # Run the agent
    asyncio.run(run_experiment(
        model=model,
        dataset_name=dataset,
        val_metric=val_metric,
        prepared_datasets_dir=paths["prepared_datasets_dir"],
        agent_datasets_dir=paths["agent_datasets_dir"],
        workspace_dir=paths["workspace_dir"],
        tags=None,
        no_root_privileges=args.root_privileges,
        iterations=iterations,
        user_prompt=args.user_prompt
    ))
    return 0
        
if __name__ == "__main__":
    sys.exit(main())