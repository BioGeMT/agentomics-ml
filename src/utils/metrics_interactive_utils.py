from rich.panel import Panel
from rich.columns import Columns
from rich.console import Console

from utils.metrics import get_classification_metrics_names, get_regression_metrics_names
from utils.user_input import get_user_input_for_int

def display_metrics_table(task_type=None):
    """Display available validation metrics in a rich table format."""
    console = Console()

    if not task_type:
        clf_metrics = get_classification_metrics_names()
        reg_metrics = get_regression_metrics_names()

        console.print(f"[bold blue]Available Validation Metrics[/bold blue]\n")

        metric_lines_clf = []
        max_num_width = len(str(len(clf_metrics) + len(reg_metrics)))

        for i, metric in enumerate(clf_metrics, 1):
            num_str = str(i).rjust(max_num_width)
            metric_lines_clf.append(f"[white]{num_str}[/white] [green]{metric}[/green]")

        metric_lines_reg = []
        for i, metric in enumerate(reg_metrics, len(clf_metrics) + 1):
            num_str = str(i).rjust(max_num_width)
            metric_lines_reg.append(f"[white]{num_str}[/white] [green]{metric}[/green]")

        boxes = [
            Panel("\n".join(metric_lines_clf), title="[bold]Metric for Classification[/bold]", border_style="cyan"),
            Panel("\n".join(metric_lines_reg), title="[bold]Metric for Regression[/bold]", border_style="yellow")
        ]

        console.print(Columns(boxes, padding=(0, 1), expand=False))
        return clf_metrics + reg_metrics

    if task_type == "classification":
        metrics_to_show = get_classification_metrics_names()
        title = "[bold]Metric for Classification[/bold]"
        border_style = "cyan"
    elif task_type == "regression":
        metrics_to_show = get_regression_metrics_names()
        title = "[bold]Metric for Regression[/bold]"
        border_style = "yellow"

    console.print(f"[bold blue]Available Validation Metrics[/bold blue]\n")

    metric_lines = []
    max_num_width = len(str(len(metrics_to_show)))

    for i, metric in enumerate(metrics_to_show, 1):
        num_str = str(i).rjust(max_num_width)
        metric_lines.append(f"[white]{num_str}[/white] [green]{metric}[/green]")

    boxes = [
        Panel("\n".join(metric_lines), title=title, border_style=border_style)
    ]

    console.print(Columns(boxes, padding=(0, 1), expand=False))
    return metrics_to_show

def interactive_metric_selection(task_type=None, default=None):
    """Get validation metric through interactive selection (requires TTY)."""
    console = Console()
    console.print("Selecting validation metric interactively...", style="cyan")
    console.print("Select validation metric for model evaluation:", style="cyan")
    showed_metrics = display_metrics_table(task_type)

    choice = get_user_input_for_int(
        prompt_text=f"Select metric number (1-{len(showed_metrics)})",
        valid_options=list(range(1, len(showed_metrics) + 1)),
    )
    
    selected_metric = showed_metrics[choice-1]
    console.print(f"Selected metric: {selected_metric}", style="green")
    return selected_metric
