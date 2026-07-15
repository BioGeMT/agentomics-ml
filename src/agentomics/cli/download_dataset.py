from __future__ import annotations

import argparse
import sys

from agentomics.datasets.create_datasets import (
    DATASET_GENERATORS,
    generate_dataset_files,
    list_dataset_names,
)


DEFAULT_DATASET = "AGO2_CLASH_Hejret2023"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download an example dataset for Agentomics.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--dataset",
        choices=sorted(DATASET_GENERATORS),
        default=DEFAULT_DATASET,
        help="Example dataset to download",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Download all available example datasets",
    )
    selection.add_argument(
        "--list",
        action="store_true",
        help="List available example datasets and exit",
    )
    return parser

def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.list:
        list_dataset_names()
        return 0

    generate_dataset_files(arguments.dataset, arguments.all)
    return 0

if __name__ == "__main__":
    sys.exit(main())
