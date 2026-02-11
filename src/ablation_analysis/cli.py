#!/usr/bin/env python3
"""CLI for ablation analysis tools."""

import argparse
from .count_runs import count_runs_by_tag
from .check_design import check_experimental_design
from .plot_cost_vs_performance import plot_cost_vs_performance
from .plot_cost_vs_performance_bars import plot_bar_charts
from .plot_cost_only import plot_cost_only
from .extract_failed_runs import extract_failed_runs


def main():
    parser = argparse.ArgumentParser(description="Ablation study analysis tools")
    parser.add_argument(
        "command",
        choices=["count", "check", "plot", "plot-bars", "plot-cost", "log-cost", "failed"],
        help="Command to run: 'count' - count runs per tag, 'check' - validate experimental design, 'plot' - create scatter plots, 'plot-bars' - create bar charts, 'plot-cost' - create cost bar charts, 'log-cost' - manually log cost to a run, 'failed' - extract failed runs to CSV"
    )

    # Add subcommand-specific arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "log-cost":
        parser.add_argument("--run", required=True, help="Run name or ID")
        parser.add_argument("--cost", type=float, required=True, help="Cost in dollars")
    if len(sys.argv) > 1 and sys.argv[1] == "failed":
        parser.add_argument("--output", default="failed_runs.csv", help="Output CSV file path")

    args = parser.parse_args()

    if args.command == "count":
        count_runs_by_tag()
    elif args.command == "check":
        check_experimental_design()
    elif args.command == "plot":
        plot_cost_vs_performance()
    elif args.command == "plot-bars":
        plot_bar_charts()
    elif args.command == "plot-cost":
        plot_cost_only()
    elif args.command == "log-cost":
        from .log_cost_manually import log_cost
        log_cost(args.run, args.cost)
    elif args.command == "failed":
        extract_failed_runs(args.output)


if __name__ == "__main__":
    main()
