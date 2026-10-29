from __future__ import annotations

from collections.abc import Mapping

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def print_metrics_table(metrics: Mapping[str, object], title: str = "Metrics") -> None:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold white",
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold cyan", no_wrap=True)
    table.add_column("Value", justify="right")

    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            rendered_value = f"{value:.6g}"
        else:
            rendered_value = str(value)
        table.add_row(str(name), rendered_value)

    Console().print(Panel(table, title=title, border_style="cyan"))
