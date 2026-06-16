import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from datasets.data_contract import (
    LABEL_COLUMN_NAME,
    METADATA_FILE_NAME,
)
from utils.task_types import TaskTypes
from utils.user_input import get_user_input_for_int

console = Console()

def _get_single_dataset_info(dataset_dir: Path) -> dict | None:
    if not dataset_dir.is_dir():
        return None

    base = {"name": dataset_dir.name, "path": dataset_dir}
    metadata_path = dataset_dir / METADATA_FILE_NAME
    if not metadata_path.is_file():
        return {**base, "status": "Unprepared"}

    try:
        metadata = json.loads(metadata_path.read_text())
        splits = metadata.get("splits", {})
        status = "Prepared" if metadata.get("prepared") is True else "Unprepared"
        return {
            **base,
            "train_rows": splits.get("train_rows"),
            "validation_rows": splits.get("validation_rows"),
            "status": status,
        }
    except Exception as exc:
        return {**base, "status": f"Invalid metadata.json: {exc}"}

def get_all_datasets_info(datasets_dir: Path) -> list[dict]:
    if not datasets_dir.exists():
        return []

    datasets_info = []
    for dataset_dir in datasets_dir.iterdir():
        dataset_info = _get_single_dataset_info(dataset_dir)
        if dataset_info:
            datasets_info.append(dataset_info)

    datasets_info.sort(key=lambda x: x["name"])
    return datasets_info

def print_datasets_table(datasets: list[dict], skip_status_column: bool = False) -> None:
    if not datasets:
        console.print("[red]No datasets found[/red]")
        return

    table = Table(border_style="cyan")
    table.add_column("#", style="dim")
    table.add_column("Dataset Name", style="white")
    table.add_column("Train", justify="right")
    table.add_column("Validation", justify="right")
    if not skip_status_column:
        table.add_column("Status")

    for i, dataset in enumerate(datasets, 1):
        train = f"{dataset['train_rows']:,}" if dataset.get('train_rows') else "[dim]-[/dim]"
        val = f"{dataset['validation_rows']:,}" if dataset.get('validation_rows') else "[dim]-[/dim]"

        row = [str(i), dataset['name'], train, val]

        if not skip_status_column:
            status = dataset["status"]
            if status == "Prepared":
                row.append(f"[green]✓ {status}[/green]")
            else:
                row.append(f"[red]⚠ {status}[/red]")

        table.add_row(*row)

    console.print(table)

def interactive_dataset_selection(datasets: list[dict]) -> str | None:
    if not datasets:
        console.print("[red]No datasets available[/red]")
        return None

    print_datasets_table(datasets, skip_status_column=True)
    dataset_indices = list(range(1, len(datasets) + 1))

    choice = get_user_input_for_int(
        "Select a dataset",
        valid_options=dataset_indices,
        default=dataset_indices[0],
    )
    selected_dataset = datasets[choice - 1]["name"]
    console.print(f"[cyan]Selected: {selected_dataset}[/cyan]")
    return selected_dataset

def select_task_type(train_labels: pd.DataFrame, interactive=False):
    if not interactive or not sys.stdin.isatty():
        raise ValueError("Task type is required. Pass --task-type classification or --task-type regression.")

    print_label_summary(train_labels)
    print("Action needed: Select the task type for this dataset.")
    print('Tip: Add "task_type" to your metadata.json to skip this prompt in the future.')

    while True:
        choice = input("Select task type ([c]lassification/[r]egression): ").strip().lower()
        if choice in ("c", "class", TaskTypes.CLASSIFICATION):
            return TaskTypes.CLASSIFICATION
        if choice in ("r", "reg", TaskTypes.REGRESSION):
            return TaskTypes.REGRESSION
        print(f"Please enter '{TaskTypes.CLASSIFICATION}' or '{TaskTypes.REGRESSION}'.")

def select_positive_class(unique_labels) -> str:
    label_options = [str(label) for label in unique_labels]
    print("Binary classification labels detected.")
    print(f"Available labels: {', '.join(label_options)}")
    print('Tip: Add "positive_class" and "negative_class" to your metadata.json to skip this prompt.')
    while True:
        positive_class = input("Enter the positive class label: ").strip()
        if positive_class in label_options:
            return positive_class
        print(f"Label '{positive_class}' not found. Please enter one of: {', '.join(label_options)}")

def select_csv_label_column(columns) -> str:
    column_options = [str(column) for column in columns]
    print("CSV dataset detected.")
    print(f"Available columns: {', '.join(column_options)}")
    print('Tip: Add "label_column" to metadata.json to skip this prompt next time.')
    while True:
        label_column = input("Enter the label column name: ").strip()
        if label_column in column_options:
            return label_column
        print(f"Column '{label_column}' not found. Please enter one of: {', '.join(column_options)}")


def print_label_summary(labels_df: pd.DataFrame) -> None:
    label_values = labels_df[LABEL_COLUMN_NAME]
    unique_values = label_values.unique()
    unique_preview = _format_unique_values_preview(unique_values)
    print(
        "\nLabel summary\n"
        f"- Column: {LABEL_COLUMN_NAME} ({label_values.dtype})\n"
        f"- Rows: {len(labels_df):,}\n"
        f"- Unique values: {len(unique_values):,}\n"
        f"- Unique labels: {unique_preview}"
    )

def _format_unique_values_preview(values, limit=20):
    preview = _format_preview_values(values[:limit])
    if len(values) > limit:
        return f"{preview}, ..."
    return preview

def _format_preview_values(values):
    formatted_values = []
    for value in values:
        if hasattr(value, "item"):
            value = value.item()
        formatted_values.append(repr(value))
    return ", ".join(formatted_values)
