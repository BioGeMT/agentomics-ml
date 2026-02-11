#!/usr/bin/env python3
"""Extract failed runs (metrics = -1) and save to CSV."""

import os
import wandb
import pandas as pd
from . import config


def extract_failed_runs(output_file="failed_runs.csv"):
    """Fetch all runs and extract failed ones (metric = -1)."""
    # Login to W&B
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    api = wandb.Api()

    print(f"Fetching runs from {config.WANDB_ENTITY}/{config.WANDB_PROJECT}...")

    # Fetch all runs with ablation tags
    filters = {"tags": {"$in": config.ABLATION_TAGS}}
    all_runs = api.runs(
        f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}",
        filters=filters
    )

    failed_runs = []
    total_fetched = 0

    for run in all_runs:
        total_fetched += 1
        try:
            # Get ablation config
            ablation_config = None
            for tag in run.tags:
                if tag.startswith("ablation:"):
                    ablation_config = tag.replace("ablation:", "")
                    break

            if not ablation_config:
                continue

            # Get summary metrics
            try:
                summary = dict(run.summary)
            except:
                summary = {}

            # Extract metrics
            test_auprc = summary.get("test_AUPRC") or summary.get("AUPRC")
            test_acc = summary.get("test_ACC") or summary.get("ACC")

            # Check if failed (metric = -1)
            is_failed = False
            if test_auprc == -1:
                is_failed = True
            if test_acc == -1:
                is_failed = True

            if is_failed:
                failed_runs.append({
                    'run_name': run.name,
                    'ablation_setup': ablation_config,
                    'failure_mode': None
                })

        except Exception as e:
            print(f"Warning: Could not parse run {run.name}: {e}")
            continue

    df = pd.DataFrame(failed_runs)

    print(f"\nTotal runs fetched: {total_fetched}")
    print(f"Failed runs found: {len(df)}")

    if len(df) > 0:
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
    else:
        print("No failed runs found.")

    return df


if __name__ == "__main__":
    extract_failed_runs()
