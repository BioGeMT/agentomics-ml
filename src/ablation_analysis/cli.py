#!/usr/bin/env python3
"""CLI for ablation analysis tools."""

import argparse
from .count_runs import count_runs_by_tag
from .check_design import check_experimental_design


def main():
    parser = argparse.ArgumentParser(description="Ablation study analysis tools")
    parser.add_argument(
        "command",
        choices=["count", "check"],
        help="Command to run: 'count' - count runs per tag, 'check' - validate experimental design"
    )

    args = parser.parse_args()

    if args.command == "count":
        count_runs_by_tag()
    elif args.command == "check":
        check_experimental_design()


if __name__ == "__main__":
    main()
