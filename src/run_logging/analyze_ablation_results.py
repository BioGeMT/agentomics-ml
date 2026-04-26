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
    parser.add_argument("--dataset", type=str, help="Filter by dataset name")
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

        # Parse config once for model/dataset extraction
        try:
            config_dict = run.config if isinstance(run.config, dict) else json.loads(run.config)
        except:
            config_dict = {}

        model_val = config_dict.get("model_name", "")
        model_name = model_val.get("value") if isinstance(model_val, dict) else model_val

        dataset_val = config_dict.get("dataset", "")
        dataset_name = dataset_val.get("value") if isinstance(dataset_val, dict) else dataset_val

        # Filter by model if specified
        if args.model and model_name != args.model:
            continue

        # Filter by dataset if specified
        if args.dataset and dataset_name != args.dataset:
            continue

        # Get ablation from tags
        ablation = "baseline"
        for tag in run.tags:
            if tag.startswith("ablation:"):
                ablation = tag.replace("ablation:", "")
                break

        # Get summary dict
        try:
            summary_dict = run.summary._json_dict if isinstance(run.summary._json_dict, dict) else json.loads(run.summary._json_dict)
        except:
            summary_dict = {}

        # Success = inference_stage == 2
        inference_stage = summary_dict.get("inference_stage", 0)
        success = (inference_stage == 2)

        # Extract specific metrics
        result = {
            "run_name": run.name,
            "model_name": model_name,
            "dataset": dataset_name,
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

    # Enforce ablation display order
    ABLATION_ORDER = [
        "baseline",
        "no_data_exploration",
        "no_data_split",
        "no_data_representation",
        "no_model_architecture",
        "no_model_training",
        "no_final_outcome",
    ]
    present = [a for a in ABLATION_ORDER if a in results_df["ablation"].unique()]
    extras = [a for a in results_df["ablation"].unique() if a not in ABLATION_ORDER]
    ordered = present + sorted(extras)
    results_df["ablation"] = pd.Categorical(results_df["ablation"], categories=ordered, ordered=True)

    # Calculate success rates
    print("\nSuccess rates by ablation:")
    summary = results_df.groupby("ablation", observed=True)["success"].agg(
        successful="sum",
        total="count",
        success_rate=lambda x: round(x.sum() / len(x) * 100, 1)
    )
    print(summary)

    # Calculate metrics stats for successful runs only
    print("\nMetrics for successful runs (mean ± std):")
    successful_runs = results_df[results_df["success"] == True]

    if len(successful_runs) > 0:
        metrics_summary = successful_runs.groupby("ablation", observed=True)[["test_ACC", "test_AUPRC", "test_F1"]].agg(["mean", "std"])
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
