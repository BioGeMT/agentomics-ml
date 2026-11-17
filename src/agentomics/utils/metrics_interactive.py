from rich.panel import Panel
from rich.columns import Columns
from rich.console import Console

from agentomics.utils.ml_tasks import get_task, get_tasks


def display_metrics_table(task_type=None):
    """Display validation metrics supplied by task implementations."""
    console = Console()
    tasks = (get_task(task_type),) if task_type else get_tasks()
    metrics_by_task = [(task, list(task.metrics())) for task in tasks]
    total_metrics = sum(len(names) for _, names in metrics_by_task)
    max_num_width = len(str(total_metrics))
    console.print("[bold blue]Available Validation Metrics[/bold blue]\n")

    boxes = []
    metrics = []
    task_colors = {task.name: color for task, color in zip(get_tasks(), ("cyan", "yellow"))}
    for task, names in metrics_by_task:
        lines = [
            f"[white]{str(i).rjust(max_num_width)}[/white] [green]{name}[/green]"
            for i, name in enumerate(names, len(metrics) + 1)
        ]
        boxes.append(Panel(
            "\n".join(lines),
            title=f"[bold]Metric for {task.name.title()}[/bold]",
            border_style=task_colors.get(task.name, "cyan"),
        ))
        metrics.extend(names)

    console.print(Columns(boxes, padding=(0, 1), expand=False))
    return metrics
