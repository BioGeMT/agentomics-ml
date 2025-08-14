import argparse
import os
import sys
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

# Add src to path so we can import our modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import utility modules
from utils.list_datasets import get_dataset_info, interactive_dataset_selection, display_datasets_table
from utils.openrouter_api import OpenRouterAPI, display_models
from utils.get_regression_candidates import get_regression_candidates

# Import the agent runner function
from run_agent import run_experiment

console = Console()

# Available validation metrics
CLASSIFICATION_METRICS = ["AUPRC", "AUROC", "ACC"]
REGRESSION_METRICS = ["MSE", "RMSE", "MAE", "R2"]
VALIDATION_METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS

def display_metrics_table(task_type=None):
    """Display available validation metrics in a rich table format."""
    if task_type:
        title = f"Available Validation Metrics for {task_type.title()} Tasks"
        if task_type == "classification":
            metrics_to_show = CLASSIFICATION_METRICS
        elif task_type == "regression":
            metrics_to_show = REGRESSION_METRICS
        else:
            metrics_to_show = VALIDATION_METRICS
    else:
        title = "Available Validation Metrics"
        metrics_to_show = VALIDATION_METRICS
    
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Metric", style="green")
    table.add_column("Description", style="blue")
    table.add_column("Task Type", style="yellow")
    
    metric_descriptions = {
        "AUPRC": "Area Under Precision-Recall Curve",
        "AUROC": "Area Under ROC Curve", 
        "ACC": "Accuracy",
        "MSE": "Mean Squared Error",
        "RMSE": "Root Mean Squared Error",
        "MAE": "Mean Absolute Error",
        "R2": "R-squared Coefficient"
    }
    
    metric_task_types = {
        "AUPRC": "Classification",
        "AUROC": "Classification", 
        "ACC": "Classification",
        "MSE": "Regression",
        "RMSE": "Regression",
        "MAE": "Regression",
        "R2": "Regression"
    }
    
    for i, metric in enumerate(metrics_to_show, 1):
        description = metric_descriptions.get(metric, "")
        task_type_desc = metric_task_types.get(metric, "")
        table.add_row(str(i), metric, description, task_type_desc)
    
    console.print(table)

def get_default_metric_for_task(task_type):
    """Get the default metric for a given task type."""
    if task_type == "classification":
        return "ACC"
    elif task_type == "regression":
        return "R2"
    else:
        return "ACC"  # Fallback default

def interactive_metric_selection(task_type=None):
    """Get validation metric through interactive selection (requires TTY)."""
    if not require_tty_for_interaction("metric selection", "Specify --val-metric argument"):
        # Return default metric for non-interactive environments
        default_metric = get_default_metric_for_task(task_type)
        console.print(f"Using default metric for {task_type or 'unknown'} task: {default_metric}", style="yellow")
        return default_metric
    
    # Determine which metrics to show based on task type
    if task_type == "classification":
        metrics_to_show = CLASSIFICATION_METRICS
        console.print("Classification dataset detected - showing classification metrics:", style="green")
    elif task_type == "regression":
        metrics_to_show = REGRESSION_METRICS
        console.print("Regression dataset detected - showing regression metrics:", style="green")
    else:
        metrics_to_show = VALIDATION_METRICS
        console.print("Task type not detected - showing all available metrics:", style="yellow")
    
    console.print("Select validation metric for model evaluation:", style="cyan")
    display_metrics_table(task_type)
    
    # Get user selection with safe prompting
    default_metric = get_default_metric_for_task(task_type)
    default_index = metrics_to_show.index(default_metric) + 1 if default_metric in metrics_to_show else 1
    
    while True:
        try:
            choice = safe_prompt(f"Select metric number (1-{len(metrics_to_show)})", str(default_index), "metric selection")
            if choice is None:
                console.print(f"Metric selection cancelled, using default: {default_metric}", style="yellow")
                return default_metric
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(metrics_to_show):
                selected_metric = metrics_to_show[choice_num - 1]
                console.print(f"Selected metric: {selected_metric}", style="green")
                return selected_metric
            else:
                console.print(f"Please enter a number between 1 and {len(metrics_to_show)}", style="red")
        except (ValueError, KeyboardInterrupt):
            console.print("Invalid input. Please enter a number.", style="red")

def check_tty_available():
    """Check if TTY is available for interactive operations."""
    return sys.stdin.isatty() and sys.stdout.isatty()

def require_tty_for_interaction(operation_name, fallback_suggestion=None):
    """Validate TTY requirement and provide helpful error messages."""
    if not check_tty_available():
        console.print(f"Interactive TTY required for {operation_name} but not available", style="red")
        console.print("This script requires an interactive terminal (TTY) for user input", style="cyan")
        if fallback_suggestion:
            console.print(f"{fallback_suggestion}", style="cyan")
        console.print("For non-interactive use, specify all required arguments explicitly", style="cyan")
        console.print("Or run from an interactive terminal session", style="cyan")
        return False
    return True

def safe_prompt(prompt_text, default=None, operation_name="input"):
    """Safely prompt for input with TTY validation."""
    if not check_tty_available():
        console.print(f"Cannot prompt for {operation_name}: no interactive TTY available", style="red")
        if default is not None:
            console.print(f"Using default: {default}", style="yellow")
            return default
        return None
    
    try:
        return Prompt.ask(prompt_text, default=default)
    except (KeyboardInterrupt, EOFError):
        console.print("\nInput cancelled", style="red")
        return None

def display_regression_columns_table(candidates):
    """Display regression target columns in a rich table format."""
    table = Table(title="Available Regression Target Columns", box=box.ROUNDED)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Column Name", style="green")
    table.add_column("Unique Values", style="blue")
    
    for i, (col_name, unique_count) in enumerate(candidates, 1):
        table.add_row(str(i), col_name, str(unique_count))
    
    console.print(table)

def print_welcome():
    """Print the welcome banner."""
    welcome_text = """Autonomous ML Agent Bot

Welcome to Agentomics-ML!
===============================================
Your Autonomous ML Agent is ready!

If you don't provide a dataset/model, you'll be prompted to select them interactively.
You can also set DATASET_NAME and MODEL_NAME as environment variables for automation."""
    
    console.print(Panel(welcome_text, style="bold blue"))
    console.print("Setting up environment...", style="blue")
    
    # Debug: Check API key status
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    wandb_key = os.getenv("WANDB_API_KEY")
    
    console.print("API Key Status:", style="cyan")
    if openrouter_key:
        console.print(f"   OPENROUTER_API_KEY: Set (length: {len(openrouter_key)} chars)", style="green")
    else:
        console.print("   OPENROUTER_API_KEY: Not set", style="red")
    
    if wandb_key:
        console.print(f"   WANDB_API_KEY: Set (length: {len(wandb_key)} chars)", style="green")
    else:
        console.print("   WANDB_API_KEY: Not set (optional for logging)", style="yellow")
    
    console.print("Environment ready!", style="green")

def get_environment_paths():
    """Get paths based on environment (Docker vs Local)."""
    if os.path.exists("/repository"):
        # Docker environment
        return {
            "datasets_dir": "/repository/datasets",
            "prepared_datasets_dir": "/repository/prepared_datasets", 
            "workspace_runs_dir": "/workspace/runs",
            "agent_datasets_dir": "/workspace/datasets"
        }
    else:
        # Local environment - use absolute paths to avoid issues when changing working directory
        base_dir = Path(__file__).parent.absolute()
        return {
            "datasets_dir": str(base_dir / "datasets"),
            "prepared_datasets_dir": str(base_dir / "prepared_datasets"),
            "workspace_runs_dir": str(base_dir / "workspace" / "runs"), 
            "agent_datasets_dir": str(base_dir / "workspace" / "datasets")
        }

def load_available_datasets(paths):
    """Load and validate available datasets."""
    datasets = get_dataset_info(paths["datasets_dir"], paths["prepared_datasets_dir"])
    if not datasets:
        console.print("No datasets available", style="red")
        return None
    return datasets

def require_datasets(paths):
    """Load datasets and exit with code 1 if none available."""
    datasets = load_available_datasets(paths)
    if not datasets:
        sys.exit(1)
    return datasets

def get_openrouter_api():
    """Get OpenRouter API instance with validation."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("OPENROUTER_API_KEY required", style="red")
        return None
    return OpenRouterAPI(api_key)

def load_available_models(limit=20):
    """Load and validate available models."""
    api = get_openrouter_api()
    if not api:
        return None
    
    models = api.get_filtered_models(limit=limit)
    if not models:
        console.print("Failed to fetch models", style="red")
        return None
    return models

def require_models(limit=20):
    """Load models and exit with code 1 if none available."""
    models = load_available_models(limit=limit)
    if not models:
        sys.exit(1)
    return models

def interactive_model_selection(limit=20):
    """Get model through interactive selection (requires TTY)."""
    if not require_tty_for_interaction("model selection", "Specify --model argument"):
        # Return default model for non-interactive environments
        default_model = "openai/gpt-4o"
        console.print(f"Using default model: {default_model}", style="yellow")
        return default_model
    
    models = load_available_models(limit=limit)
    
    if models:
        console.print("Showing coding & reasoning models available with your OpenRouter API key", style="cyan")
        display_models(models)
        
        # Get user selection with safe prompting
        while True:
            try:
                choice = safe_prompt(f"Select model number (1-{len(models)})", "1", "model selection")
                if choice is None:
                    console.print("Model selection cancelled", style="red")
                    return "openai/gpt-4o"
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]['id']
                    console.print(f"Selected model: {selected_model}", style="green")
                    return selected_model
                else:
                    console.print(f"Please enter a number between 1 and {len(models)}", style="red")
            except (ValueError, KeyboardInterrupt):
                console.print("Invalid input. Please enter a number.", style="red")
    else:
        console.print("No models available, using default", style="yellow")
        return "openai/gpt-4o"

def run_agent(dataset: str, model: str, **kwargs):
    """Run the agent with the selected parameters."""
    console.print("\n" + "━" * 80, style="cyan")
    console.print("Launching Agentomics-ML", style="green")
    console.print("━" * 80, style="cyan")
    
    # Display configuration
    console.print(f"Dataset: {dataset}", style="cyan")
    console.print(f"Model: {model}", style="yellow")
    for key, value in kwargs.items():
        if value:
            console.print(f"{key.replace('_', ' ').title()}: {value}", style="purple")
    console.print("━" * 80, style="cyan")
    
    # Get environment paths
    paths = get_environment_paths()
    
    # Build arguments for direct function call
    args = [
        "--dataset-name", dataset,
        "--model", model,
        "--prepared-datasets-dir", paths["prepared_datasets_dir"],
        "--agent-datasets-dir", paths["agent_datasets_dir"],
        "--workspace-dir", paths["workspace_runs_dir"],
        "--no-root-privileges"
    ]
    
    # Add additional arguments (skip target_column as it's handled in dataset preparation)
    for key, value in kwargs.items():
        if value and key != 'target_column':
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Execute the agent directly
    try:
        console.print(f"Running agent with args: {' '.join(args)}", style="dim")
        
        # Change to src directory for proper module loading
        original_cwd = os.getcwd()
        src_dir = Path(__file__).parent / "src"
        os.chdir(src_dir)
        
        # Mock sys.argv for argparse
        original_argv = sys.argv
        sys.argv = ["run_agent.py"] + args
        
        try:
            # Call the agent experiment runner (async function)
            asyncio.run(run_experiment())
            return 0
        finally:
            # Restore original state
            sys.argv = original_argv
            os.chdir(original_cwd)
            
    except Exception as e:
        console.print(f"Error running agent: {e}", style="red")
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agentomics-ML Entry Point")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")
    parser.add_argument("--list-metrics", action="store_true", help="List available validation metrics and exit")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--target-column", help="Target column for regression")
    parser.add_argument("--val-metric", help="Validation metric", choices=VALIDATION_METRICS)
    
    args, unknown_args = parser.parse_known_args()
    paths = get_environment_paths()
    
    # Handle list-only modes (these don't require interactivity)
    if args.list_datasets:
        console.print("Available Datasets", style="cyan")
        datasets = load_available_datasets(paths)
        if datasets:
            display_datasets_table(datasets)
        else:
            console.print("No datasets found", style="yellow")
        return 0
    
    if args.list_models:
        console.print("Available Coding & Reasoning Models", style="cyan")
        models = require_models(limit=30)
        display_models(models)
        return 0
    
    if args.list_metrics:
        console.print("Available Validation Metrics", style="cyan")
        display_metrics_table()  # Show all metrics when listing
        return 0
    
    # Get dataset and model from args or environment
    dataset = args.dataset or os.getenv("DATASET_NAME")
    model = args.model or os.getenv("MODEL_NAME")
    
    # For interactive mode (when dataset/model missing), require TTY
    if not dataset or not model:
        if not check_tty_available():
            console.print("Interactive TTY required for dataset/model selection but not available", style="red")
            console.print("For non-interactive use, specify both --dataset and --model arguments", style="cyan")
            console.print("Or use the shell script: ./run.sh with --dataset and --model flags", style="cyan")
            console.print("Example: python agentomics-entrypoint.py --dataset heart_disease --model 'openai/gpt-4'", style="cyan")
            return 1
    
    # Print welcome and show mode
    print_welcome()
    if os.path.exists("/repository"):
        console.print("Running in DOCKER mode", style="blue")
    else:
        console.print("Running in LOCAL mode", style="blue")
    
    # Interactive dataset selection if needed (TTY is guaranteed at this point)
    if not dataset:
        console.print("Selecting dataset interactively...", style="cyan")
        datasets = require_datasets(paths)
        
        dataset = interactive_dataset_selection(datasets)
        if not dataset:
            console.print("No dataset selected", style="red")
            return 1
    
    # Handle regression target selection right after dataset selection
    target_column = args.target_column or os.getenv("TARGET_COLUMN")
    if not target_column and dataset:
        console.print("Checking for regression target selection...", style="cyan")
        
        # First detect if this is actually a regression dataset by checking the task type
        try:
            from src.utils.list_datasets import detect_task_type
            task_type = detect_task_type(f"{paths['prepared_datasets_dir']}/{dataset}")
            console.print(f"Detected task type: {task_type}", style="dim")
        except Exception:
            task_type = "Unknown"
        
        # Only ask for target column selection if it's actually a regression task
        if task_type == "regression":
            candidates = get_regression_candidates(f"{paths['prepared_datasets_dir']}/{dataset}")
            
            if candidates:
                console.print("Regression dataset detected! Please select target column:", style="green")
            
            if len(candidates) == 1:
                target_column = candidates[0][0]
                console.print(f"Auto-selected only available regression target: {target_column}", style="green")
            elif len(candidates) > 1:
                # Check if TTY is available for multi-column regression target selection
                if not check_tty_available():
                    console.print("Multiple regression targets available but no TTY for selection", style="red")
                    console.print("Specify --target-column argument for non-interactive use", style="cyan")
                    display_regression_columns_table(candidates)
                    return 1
                
                if not require_tty_for_interaction("regression target column selection", "Specify --target-column argument"):
                    # Use first candidate as default
                    target_column = candidates[0][0]
                    console.print(f"Using default target column: {target_column}", style="yellow")
                else:
                    display_regression_columns_table(candidates)
                    
                    # Interactive target column selection with safe prompting
                    while True:
                        try:
                            choice = safe_prompt(f"Select target column (1-{len(candidates)})", "1", "target column selection")
                            if choice is None:
                                console.print("Target column selection cancelled, using first available", style="yellow")
                                target_column = candidates[0][0]
                                break
                            
                            choice_num = int(choice)
                            if 1 <= choice_num <= len(candidates):
                                target_column = candidates[choice_num - 1][0]
                                console.print(f"Selected target column: {target_column}", style="green")
                                break
                            else:
                                console.print(f"Please enter a number between 1 and {len(candidates)}", style="red")
                        except (ValueError, KeyboardInterrupt):
                            console.print("Invalid input. Please enter a number.", style="red")
        else:
            # Not a regression task, skip target column selection entirely
            console.print(f"{task_type.title()} dataset - no target column selection needed", style="green")
    
    # Interactive metric selection if needed (TTY is guaranteed at this point)
    # Note: Metric selection comes before model selection to help users choose the right model for their optimization goal
    val_metric = args.val_metric or os.getenv("VAL_METRIC")
    if not val_metric:
        console.print("Selecting validation metric interactively...", style="cyan")
        val_metric = interactive_metric_selection(task_type) # Pass task_type here
    
    # Interactive model selection if needed (TTY is guaranteed at this point)
    if not model:
        console.print("Selecting model interactively...", style="cyan")
        model = interactive_model_selection(limit=50)
    
    # Run the agent
    return run_agent(
        dataset=dataset,
        model=model,
        target_column=target_column,
        val_metric=val_metric
    )

if __name__ == "__main__":
    sys.exit(main())