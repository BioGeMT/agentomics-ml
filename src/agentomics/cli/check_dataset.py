from __future__ import annotations

import argparse
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

def _print_summary(summary: dict) -> None:
    console.print("✓ Dataset is valid", style="green")
    console.print(f"Task type: {summary.get('task_type')}")

    splits = summary.get("splits", {})
    console.print(f"Train rows: {splits.get('train_rows')}")
    validation_rows = splits.get("validation_rows")
    if validation_rows is not None:
        console.print(f"Validation rows: {validation_rows}")
    else:
        console.print("Validation split: not provided (the agent will create one)")

    test_rows = summary.get("test_rows")
    if test_rows is not None:
        console.print(f"Test rows: {test_rows}")

    if "label_to_scalar" in summary:
        console.print(f"Label mapping: {summary['label_to_scalar']}")
    console.print(f"Input structure: {summary.get('input_structure')}")

def main() -> int:
    arguments = build_parser().parse_args()
    try:
        summary = check_dataset(source_dir=arguments.dataset_dir)
    except (ValueError, OSError) as error: # ValueError = data-contract violation; OSError = missing/unreadable paths.
        console.print(f"✗ Dataset is not valid: {error}", style="red")
        return 1
    _print_summary(summary)
    return 0

if __name__ == "__main__":
    sys.exit(main())
