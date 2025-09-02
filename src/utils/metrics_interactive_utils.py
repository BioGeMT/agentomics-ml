from rich.table import Table
from rich import box
from rich.console import Console

from eval.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.user_input import get_user_input_for_int

def display_metrics_table(task_type=None):
    """Display available validation metrics in a rich table format."""
    console = Console()

    if(not task_type):
        # Displaying all metrics
        clf_metrics = get_classification_metrics_names()
        reg_metrics = get_regression_metrics_names()

        clf_metric_task_types = {metric : "Classification" for metric in clf_metrics}
        reg_metric_task_types = {metric : "Regression" for metric in reg_metrics}

        metrics_to_show = clf_metrics + reg_metrics
        metric_task_types = {**clf_metric_task_types, **reg_metric_task_types}
    if(task_type == "classification"):
        metrics_to_show = get_classification_metrics_names()
        metric_task_types = {metric : "Classification" for metric in metrics_to_show}
    if(task_type == "regression"):
        metrics_to_show = get_regression_metrics_names()
        metric_task_types = {metric : "Regression" for metric in metrics_to_show}

    title = "Available Validation Metrics"
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Metric", style="green")
    table.add_column("Task Type", style="yellow")
    
    for i, metric in enumerate(metrics_to_show, 1):
        table.add_row(str(i), metric, metric_task_types[metric])
    
    console.print(table)
    return metrics_to_show

def interactive_metric_selection(task_type=None):
    """Get validation metric through interactive selection (requires TTY)."""
    console = Console()
    console.print("Selecting validation metric interactively...", style="cyan")
    console.print("Select validation metric for model evaluation:", style="cyan")
    showed_metrics = display_metrics_table(task_type)

    choice = get_user_input_for_int(
        prompt_text=f"Select metric number (1-{len(showed_metrics)})",
        valid_options=list(range(1, len(showed_metrics) + 1)),
    )
    if not choice:
        console.print(f"Metric selection cancelled, using default.", style="yellow")
        return None
    
    selected_metric = showed_metrics[choice-1]
    console.print(f"Selected metric: {selected_metric}", style="green")
    return selected_metric
