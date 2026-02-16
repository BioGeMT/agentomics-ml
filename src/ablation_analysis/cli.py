#!/usr/bin/env python3
"""CLI for ablation analysis tools."""

import os
import sys
import argparse
from .count_runs import count_runs_by_tag
from .check_design import check_experimental_design
from .list_runs import list_runs
from .extract_architectures import extract_architectures, DEFAULT_RESULTS_DIR
from .categorize import categorize_architectures, DEFAULT_MODEL, DEFAULT_DELAY
from .plot_architectures import plot_architectures
from .plot_cost_vs_performance import plot_cost_vs_performance
from .plot_cost_vs_performance_bars import plot_bar_charts
from .plot_cost_only import plot_cost_only
from .extract_failed_runs import extract_failed_runs
from .plot_failures import plot_failures
from . import config


def main():
    parser = argparse.ArgumentParser(description="Ablation study analysis tools")
    COMMANDS = [
        "count", "check", "list-runs", "extract-arch", "categorize",
        "plot-arch", "plot-failures", "plot", "plot-bars", "plot-cost", "log-cost", "failed",
    ]
    parser.add_argument("command", choices=COMMANDS, help="Command to run")

    # Add subcommand-specific arguments
    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "log-cost":
        parser.add_argument("--run", required=True, help="Run name or ID")
        parser.add_argument("--cost", type=float, required=True, help="Cost in dollars")
    if cmd == "list-runs":
        parser.add_argument("--output", type=str, default=None,
                            help="Save run names to this file (one per line)")
    if cmd == "extract-arch":
        parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                            help=f"Path to results directory (default: {DEFAULT_RESULTS_DIR})")
        parser.add_argument("--output", type=str,
                            default=os.path.join(config.OUTPUT_DIR, "architectures.csv"),
                            help="Output CSV file path")
    if cmd == "categorize":
        parser.add_argument("--input", type=str,
                            default=os.path.join(config.OUTPUT_DIR, "architectures.csv"),
                            help="Input CSV file from extract-arch")
        parser.add_argument("--output", type=str,
                            default=os.path.join(config.OUTPUT_DIR, "architectures_categorized.csv"),
                            help="Output CSV file with categories")
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                            help=f"LLM model for categorization (default: {DEFAULT_MODEL})")
        parser.add_argument("--delay", type=int, default=DEFAULT_DELAY,
                            help=f"Delay in seconds between API requests (default: {DEFAULT_DELAY})")
    if cmd == "plot-arch":
        parser.add_argument("--input", type=str,
                            default=os.path.join(config.OUTPUT_DIR, "architectures_categorized.csv"),
                            help="Input categorized CSV file")
        parser.add_argument("--output-dir", type=str,
                            default=config.OUTPUT_DIR,
                            help="Output directory for plots")
    if cmd == "plot-failures":
        parser.add_argument("--input", type=str,
                            default=os.path.join(config.OUTPUT_DIR, "failed_with_model.csv"),
                            help="Input CSV with failure_category column")
        parser.add_argument("--output-dir", type=str,
                            default=config.OUTPUT_DIR,
                            help="Output directory for plots")
    if cmd == "failed":
        parser.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "failed_runs.csv"),
                            help="Output CSV file path")

    args = parser.parse_args()

    if args.command == "list-runs":
        list_runs(output_file=getattr(args, 'output', None))
    elif args.command == "count":
        count_runs_by_tag()
    elif args.command == "check":
        check_experimental_design()
    elif args.command == "extract-arch":
        extract_architectures(
            results_dir=args.results_dir,
            output_file=args.output,
        )
    elif args.command == "categorize":
        categorize_architectures(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
            delay=args.delay,
        )
    elif args.command == "plot-arch":
        plot_architectures(
            input_file=args.input,
            output_dir=args.output_dir,
        )
    elif args.command == "plot-failures":
        plot_failures(
            input_file=args.input,
            output_dir=args.output_dir,
        )
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
