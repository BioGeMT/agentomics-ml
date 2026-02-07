#!/usr/bin/env python3
"""Plot Cost vs Performance for each dataset × model combination."""

import os
import json
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from . import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# Models with cost data (skip gpt-oss:20b as it was run locally)
MODELS_WITH_COST = [
    "google/gemini-3-flash-preview",
    "x-ai/grok-4.1-fast"
]

# Performance metrics per dataset
DATASET_METRICS = {
    "AGO2_CLASH_Hejret": "AUPRC",
    "human_enhancers_cohn": "ACC"
}

# Ablation colors and markers
ABLATION_COLORS = {
    "baseline": "#1f77b4",
    "no_all_steps": "#bcbd22",
    "no_data_exploration": "#ff7f0e",
    "no_data_representation": "#d62728",
    "no_data_split": "#2ca02c",
    "no_final_outcome": "#e377c2",
    "no_model_architecture": "#9467bd",
    "no_model_training": "#8c564b",
}

ABLATION_MARKERS = {
    "baseline": "o",
    "no_all_steps": "h",
    "no_data_exploration": "s",
    "no_data_representation": "D",
    "no_data_split": "^",
    "no_final_outcome": "*",
    "no_model_architecture": "v",
    "no_model_training": "P",
}

ABLATION_LABELS = {
    "baseline": "Baseline",
    "no_all_steps": "No All Steps",
    "no_data_exploration": "No Data Exploration",
    "no_data_representation": "No Data Representation",
    "no_data_split": "No Data Split",
    "no_final_outcome": "No Final Outcome",
    "no_model_architecture": "No Model Architecture",
    "no_model_training": "No Model Training",
}


def fetch_runs_data():
    """Fetch all runs and extract relevant data."""
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

    runs_data = []
    total_fetched = 0
    skipped_no_ablation = 0
    skipped_wrong_model = 0
    skipped_no_metrics = 0
    skipped_failed = 0
    skipped_no_cost = 0

    # Debug: track runs per combination
    debug_runs = defaultdict(list)

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
                skipped_no_ablation += 1
                continue

            # Get config
            if isinstance(run.config, str):
                cfg = json.loads(run.config)
            else:
                cfg = dict(run.config)

            # Extract dataset
            dataset = None
            if "dataset" in cfg:
                val = cfg["dataset"]
                dataset = val.get("value") if isinstance(val, dict) else val

            # Extract model
            model = None
            if "model_name" in cfg:
                val = cfg["model_name"]
                model = val.get("value") if isinstance(val, dict) else val
            elif "model" in cfg:
                val = cfg["model"]
                model = val.get("value") if isinstance(val, dict) else val

            # Skip if not one of the models with cost data
            if model not in MODELS_WITH_COST:
                skipped_wrong_model += 1
                continue

            if not dataset or not model:
                continue

            # Get summary metrics
            try:
                summary = dict(run.summary)
            except:
                summary = {}

            # Extract metrics (try both with and without test_ prefix)
            test_auprc = summary.get("test_AUPRC") or summary.get("AUPRC")
            test_acc = summary.get("test_ACC") or summary.get("ACC")

            # Skip failed runs (metric = -1)
            is_failed = False
            if test_auprc == -1:
                test_auprc = None
                is_failed = True
            if test_acc == -1:
                test_acc = None
                is_failed = True

            if is_failed:
                skipped_failed += 1

            # Get cost from api_usage/usage (following analyze_costs.py logic)
            total_cost = None

            # First try direct key with slash
            if "api_usage/usage" in summary:
                total_cost = summary.get("api_usage/usage")

            # Then try nested dict
            if total_cost is None:
                api_usage = summary.get("api_usage", {})

                if isinstance(api_usage, dict):
                    # Try api_usage.usage.cost or api_usage.usage (if usage IS the cost)
                    usage = api_usage.get("usage")
                    if isinstance(usage, (int, float)):
                        # usage is the cost value directly
                        total_cost = usage
                    elif isinstance(usage, dict):
                        # usage is a dict with cost inside
                        total_cost = usage.get("cost", usage.get("total_cost"))

                    # Fallback: try direct api_usage.cost
                    if total_cost is None:
                        total_cost = api_usage.get("cost", api_usage.get("total_cost"))

                # Try top-level summary
                if total_cost is None:
                    total_cost = summary.get("total_cost", summary.get("api_cost"))

            # If still not found, try history (Charts data)
            if total_cost is None:
                try:
                    history = run.history(keys=["api_usage/usage"], samples=10000)
                    if not history.empty and "api_usage/usage" in history.columns:
                        usage_values = history["api_usage/usage"].dropna()
                        if len(usage_values) > 0:
                            total_cost = usage_values.iloc[-1]
                except:
                    pass

            # Check if run was successful (has metrics)
            if test_auprc is None and test_acc is None:
                skipped_no_metrics += 1
                debug_runs[f"{dataset}_{model}"].append(f"{run.name}: NO METRICS")
                continue

            # Check if we have cost data
            if total_cost is None:
                skipped_no_cost += 1
                debug_runs[f"{dataset}_{model}"].append(f"{run.name}: NO COST (has metrics)")
                continue

            # Track successful runs
            debug_runs[f"{dataset}_{model}"].append(f"{run.name}: OK")

            runs_data.append({
                'run_id': run.id,
                'run_name': run.name,
                'dataset': dataset,
                'model': model,
                'ablation': ablation_config,
                'test_AUPRC': test_auprc,
                'test_ACC': test_acc,
                'total_cost': total_cost
            })

        except Exception as e:
            print(f"Warning: Could not parse run {run.name}: {e}")
            continue

    df = pd.DataFrame(runs_data)

    print(f"\nDebug info:")
    print(f"  Total runs fetched: {total_fetched}")
    print(f"  Skipped (no ablation tag): {skipped_no_ablation}")
    print(f"  Skipped (wrong model): {skipped_wrong_model}")
    print(f"  Skipped (failed, metric=-1): {skipped_failed}")
    print(f"  Skipped (no metrics): {skipped_no_metrics}")
    print(f"  Skipped (no cost data): {skipped_no_cost}")
    print(f"  Final runs with data: {len(df)}\n")

    # Show detailed breakdown for each combination
    print("Detailed breakdown by dataset × model:")
    for key, runs_list in sorted(debug_runs.items()):
        print(f"\n{key}:")
        ok_runs = [r for r in runs_list if "OK" in r]
        problem_runs = [r for r in runs_list if "OK" not in r]
        print(f"  OK: {len(ok_runs)}")
        if problem_runs:
            print(f"  Issues: {len(problem_runs)}")
            for run in problem_runs:  # Show all
                print(f"    - {run}")

    return df


def create_combined_plot(df, dataset, output_dir="ablation_res"):
    """Create cost vs performance scatter plot combining both models for a dataset."""

    # Filter data for this dataset
    data = df[df['dataset'] == dataset].copy()

    if len(data) == 0:
        print(f"Warning: No data for {dataset}")
        return

    # Get performance metric for this dataset
    perf_metric = DATASET_METRICS.get(dataset)
    if not perf_metric:
        print(f"Warning: No metric defined for {dataset}")
        return

    perf_col = f"test_{perf_metric}"

    # Filter out rows with missing data
    data = data.dropna(subset=[perf_col, 'total_cost'])

    if len(data) == 0:
        print(f"Warning: No valid data for {dataset}")
        return

    models = sorted(data['model'].unique())
    print(f"\nCreating combined plot for {dataset}")
    print(f"  Models: {models}")
    print(f"  Runs: {len(data)}")
    print(f"  Ablations: {sorted(data['ablation'].unique())}")

    # Calculate means and std for each (model, ablation) combination
    aggregated = data.groupby(['model', 'ablation']).agg({
        'total_cost': ['mean', 'std'],
        perf_col: ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated.columns = ['model', 'ablation', 'cost_mean', 'cost_std', 'perf_mean', 'perf_std']

    # Replace NaN std with 0 (for cases with only 1 replicate)
    aggregated['cost_std'].fillna(0, inplace=True)
    aggregated['perf_std'].fillna(0, inplace=True)

    print(f"  Aggregated to {len(aggregated)} mean points")

    # Create figure (larger for better readability)
    fig, ax = plt.subplots(figsize=(14, 11))

    # Plot each model with different fill style
    ablations = sorted(data['ablation'].unique())

    for ablation in ablations:
        color = ABLATION_COLORS.get(ablation, "#000000")
        marker = ABLATION_MARKERS.get(ablation, "o")
        label = ABLATION_LABELS.get(ablation, ablation.replace("_", " ").title())

        for i, model in enumerate(models):
            agg_data = aggregated[(aggregated['ablation'] == ablation) & (aggregated['model'] == model)]

            if len(agg_data) == 0:
                continue

            row = agg_data.iloc[0]

            # First model: filled markers
            # Second model: hollow markers
            if i == 0:  # First model (gemini) - filled
                ax.errorbar(
                    row['cost_mean'],
                    row['perf_mean'],
                    xerr=row['cost_std'],
                    yerr=row['perf_std'],
                    fmt=marker,
                    color=color,
                    markersize=14,
                    markeredgecolor='black',
                    markeredgewidth=2,
                    capsize=5,
                    capthick=2,
                    elinewidth=2,
                    alpha=0.8,
                    label=label if i == 0 else None
                )
            else:  # Second model (grok) - hollow
                ax.errorbar(
                    row['cost_mean'],
                    row['perf_mean'],
                    xerr=row['cost_std'],
                    yerr=row['perf_std'],
                    fmt=marker,
                    markerfacecolor='none',
                    markeredgecolor=color,
                    markersize=14,
                    markeredgewidth=2.5,
                    capsize=5,
                    capthick=2,
                    elinewidth=2,
                    alpha=0.8
                )

    # Customize plot
    ax.set_xlabel("Run cost ($)", fontsize=18, fontweight='bold')
    ax.set_ylabel(perf_metric, fontsize=18, fontweight='bold')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Create custom legend with model differentiation
    from matplotlib.lines import Line2D

    # Ablation legend entries
    legend_elements = []
    for ablation in ablations:
        color = ABLATION_COLORS.get(ablation, "#000000")
        marker = ABLATION_MARKERS.get(ablation, "o")
        label = ABLATION_LABELS.get(ablation, ablation.replace("_", " ").title())
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                   markeredgecolor='black', markersize=10, label=label, linewidth=0)
        )

    # Add model differentiation
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                   markeredgecolor='black', markersize=10,
                                   label=f'{models[0].split("/")[-1]} (filled)', linewidth=0))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                   markeredgecolor='gray', markersize=10,
                                   label=f'{models[1].split("/")[-1]} (hollow)', linewidth=0, markeredgewidth=2))

    ax.legend(
        handles=legend_elements,
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12
    )

    plt.tight_layout()

    # Save plot as PDF only (vector format, best quality)
    os.makedirs(output_dir, exist_ok=True)
    safe_dataset = dataset.replace('_', '-')
    output_file = os.path.join(output_dir, f"cost_vs_perf_{safe_dataset}_combined.pdf")

    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    plt.close()


def create_plot(df, dataset, model, output_dir="ablation_res"):
    """Create cost vs performance scatter plot for a dataset × model combination."""

    # Filter data
    data = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()

    if len(data) == 0:
        print(f"Warning: No data for {dataset} × {model}")
        return

    # Get performance metric for this dataset
    perf_metric = DATASET_METRICS[dataset]
    perf_col = f"test_{perf_metric}"

    # Check if we have the metric
    if data[perf_col].isna().all():
        print(f"Warning: No {perf_metric} data for {dataset} × {model}")
        return

    # Filter out rows with missing data
    data = data.dropna(subset=[perf_col, 'total_cost'])

    if len(data) == 0:
        print(f"Warning: No valid data for {dataset} × {model}")
        return

    print(f"\nCreating plot for {dataset} × {model}")
    print(f"  Runs: {len(data)}")
    print(f"  Ablations: {sorted(data['ablation'].unique())}")

    # Create figure (larger for better readability when scaled down)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each ablation - individual points only
    ablations = sorted(data['ablation'].unique())

    for ablation in ablations:
        ablation_data = data[data['ablation'] == ablation]

        color = ABLATION_COLORS.get(ablation, "#000000")
        marker = ABLATION_MARKERS.get(ablation, "o")
        label = ABLATION_LABELS.get(ablation, ablation.replace("_", " ").title())

        # Plot individual points
        ax.scatter(
            ablation_data['total_cost'],
            ablation_data[perf_col],
            c=color,
            marker=marker,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            label=label
        )

    # Customize plot
    ax.set_xlabel("Run cost ($)", fontsize=18, fontweight='bold')
    ax.set_ylabel(perf_metric, fontsize=18, fontweight='bold')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend with larger font
    ax.legend(
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
        markerscale=1.2
    )

    plt.tight_layout()

    # Save plot as PDF only (vector format, best quality)
    os.makedirs(output_dir, exist_ok=True)
    safe_dataset = dataset.replace('_', '-')
    safe_model = model.replace('/', '_')
    output_file = os.path.join(output_dir, f"cost_vs_perf_{safe_dataset}_{safe_model}.pdf")

    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    plt.close()


def plot_cost_vs_performance(output_dir="ablation_res"):
    """Create cost vs performance plots combining models per dataset."""

    print("="*70)
    print("COST VS PERFORMANCE ANALYSIS (Combined Models)")
    print("="*70)

    # Fetch data
    df = fetch_runs_data()

    if len(df) == 0:
        print("No data found!")
        return

    # Get unique datasets and models
    datasets = sorted(df['dataset'].unique())
    models = sorted(df['model'].unique())

    print(f"\nDatasets: {datasets}")
    print(f"Models: {models}")

    # Create one combined plot per dataset
    for dataset in datasets:
        create_combined_plot(df, dataset, output_dir)

    print("\n" + "="*70)
    print("Done!")


if __name__ == "__main__":
    plot_cost_vs_performance()
