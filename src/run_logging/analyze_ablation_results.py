#!/usr/bin/env python3
"""
Simple Ablation Analysis - Calculate success rates from W&B

Usage:
    python analyze_ablation_results.py --tags ablation_study_test_friday
"""

import argparse
import os
import json
import pandas as pd
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", default="agentomics_ablation")
    parser.add_argument("--wandb-entity", default="ceitec-ai")
    parser.add_argument("--tags", nargs="+", help="Filter by tags (optional)")
    parser.add_argument("--after", type=str, help="Filter runs after this date (YYYY-MM-DD HH:MM format, e.g., '2025-10-17 20:00')")
    parser.add_argument("--model", type=str, help="Filter by model name (e.g., 'gpt-oss:20b')")
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    # Login
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    # Fetch runs
    print(f"Fetching runs from {args.wandb_entity}/{args.wandb_project}...")
    api = wandb.Api()

    # Build filters
    filters = {}
    if args.tags:
        filters["tags"] = {"$in": args.tags}

    runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}", filters=filters if filters else None)

    # Parse date filter if provided
    from datetime import datetime
    after_date = None
    if args.after:
        after_date = datetime.strptime(args.after, "%Y-%m-%d %H:%M")
        print(f"Filtering runs after: {after_date}")

    # Collect all runs first (as per W&B docs pattern)
    all_runs = []
    for run in runs:
        # Filter by date if specified
        if after_date:
            # Parse W&B timestamp (format: 2025-10-17T20:00:00Z)
            run_date_str = run.created_at.replace('Z', '')
            run_date = datetime.fromisoformat(run_date_str)
            if run_date <= after_date:
                continue

        # Filter by model if specified
        if args.model:
            try:
                # run.config is a JSON string, parse it
                config_dict = json.loads(run.config)
                model_name = config_dict.get("model_name", {}).get("value", "")
                if model_name != args.model:
                    continue
            except:
                # Skip runs we can't parse
                continue

        # Get ablation from tags
        ablation = "baseline"
        for tag in run.tags:
            if tag.startswith("ablation:"):
                ablation = tag.replace("ablation:", "")
                break

        # Parse the JSON string from _json_dict
        try:
            summary_dict = json.loads(run.summary._json_dict)
        except:
            summary_dict = {}

        # Success = inference_stage == 2
        inference_stage = summary_dict.get("inference_stage", 0)
        success = (inference_stage == 2)

        # Extract specific metrics
        result = {
            "run_name": run.name,
            "ablation": ablation,
            "success": success,
            "inference_stage": inference_stage,
            "test_ACC": summary_dict.get("ACC"),
            "test_AUPRC": summary_dict.get("AUPRC"),
            "test_F1": summary_dict.get("F1"),
        }

        all_runs.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_runs)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(results_df)} runs to {args.output}")

    # Check if we have any runs
    if len(results_df) == 0:
        print("No runs found matching the filters")
        return

    # Calculate success rates
    print("\nSuccess rates by ablation:")
    summary = results_df.groupby("ablation")["success"].agg(
        successful="sum",
        total="count",
        success_rate=lambda x: round(x.sum() / len(x) * 100, 1)
    )
    print(summary)

    # Calculate metrics stats for successful runs only
    print("\nMetrics for successful runs (mean ± std):")
    successful_runs = results_df[results_df["success"] == True]

    if len(successful_runs) > 0:
        metrics_summary = successful_runs.groupby("ablation")[["test_ACC", "test_AUPRC", "test_F1"]].agg(["mean", "std"])
        metrics_summary = metrics_summary.round(4)
        print(metrics_summary)

        # Save metrics summary to separate CSV
        metrics_output = args.output.replace(".csv", "_metrics_summary.csv")
        metrics_summary.to_csv(metrics_output)
        print(f"\nMetrics summary saved to: {metrics_output}")
    else:
        print("No successful runs found")


if __name__ == "__main__":
    main()
