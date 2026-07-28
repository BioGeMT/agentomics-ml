from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from agentomics.datasets.dataset_preparation import check_dataset

console = Console()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that a dataset matches the Agentomics run contract, "
            "without launching a run."
        ),
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory to validate",
    )
    return parser

def _is_tty_available() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _suggested_metadata(summary: dict) -> dict:
    applicable_keys = (
        "task_type",
        "label_column",
        "id_column",
        "label_to_scalar",
    )
    return {
        key: summary[key]
        for key in applicable_keys
        if key in summary
    }


def _print_summary(summary: dict, interactive: bool) -> None:
    console.print("✓ Dataset is valid", style="green")
    console.print(f"Task type: {summary.get('task_type')}")

    splits = summary.get("splits", {})
    console.print(f"Train rows: {splits.get('train_rows')}")
    validation_rows = splits.get("validation_rows")
    if validation_rows is not None:
        console.print(f"Validation rows: {validation_rows}")
    else:
        console.print("Validation split: not provided (the agent will create one)")

    for split_name, row_count in summary.get("test_splits", {}).items():
        console.print(f"{split_name} rows: {row_count}")

    if "label_to_scalar" in summary:
        console.print(f"Label mapping: {summary['label_to_scalar']}")
    console.print(f"Input structure: {summary.get('input_structure')}")

    if interactive:
        console.print("Suggested metadata.json:", style="cyan")
        console.print(json.dumps(_suggested_metadata(summary), indent=4))

def main() -> int:
    arguments = build_parser().parse_args()
    interactive = _is_tty_available()
    try:
        summary = check_dataset(
            source_dir=arguments.dataset_dir,
            interactive=interactive,
        )
    except (ValueError, OSError) as error: # ValueError = data-contract violation; OSError = missing/unreadable paths.
        console.print(f"✗ Dataset is not valid: {error}", style="red")
        return 1
    _print_summary(summary, interactive)
    return 0

if __name__ == "__main__":
    sys.exit(main())
