#!/usr/bin/env python3
"""Create bar chart showing performance and cost for ablation study."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_cost_vs_performance import fetch_runs_data, DATASET_METRICS

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# Colors for models
MODEL_COLORS = {
    'google/gemini-3-flash-preview': '#1f77b4',  # blue
    'x-ai/grok-4.1-fast': '#ff7f0e',  # orange
}


def create_bar_chart(df, dataset, output_dir="ablation_res"):
    """Create bar chart with performance bars and cost line."""

    # Filter data for this dataset
    data = df[df['dataset'] == dataset].copy()

    if len(data) == 0:
        print(f"Warning: No data for {dataset}")
        return

    # Get performance metric
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
    print(f"\nCreating bar chart for {dataset}")
    print(f"  Models: {models}")
    print(f"  Runs: {len(data)}")

    # Calculate means and std for each (model, ablation) combination
    aggregated = data.groupby(['model', 'ablation']).agg({
        'total_cost': ['mean', 'std'],
        perf_col: ['mean', 'std']
    }).reset_index()

    aggregated.columns = ['model', 'ablation', 'cost_mean', 'cost_std', 'perf_mean', 'perf_std']
    aggregated['cost_std'].fillna(0, inplace=True)
    aggregated['perf_std'].fillna(0, inplace=True)

    # Get ablations
    ablations = sorted(data['ablation'].unique())

    # Create figure
    fig, ax1 = plt.subplots(figsize=(16, 10))

    # Set up bar positions
    x = np.arange(len(ablations))
    width = 0.35

    # Plot performance bars
    for i, model in enumerate(models):
        model_data = aggregated[aggregated['model'] == model]

        perf_means = []
        perf_stds = []
        for ablation in ablations:
            row = model_data[model_data['ablation'] == ablation]
            if len(row) > 0:
                perf_means.append(row['perf_mean'].values[0])
                perf_stds.append(row['perf_std'].values[0])
            else:
                perf_means.append(0)
                perf_stds.append(0)

        offset = width * (i - 0.5)
        model_label = model.split('/')[-1]
        color = MODEL_COLORS.get(model, '#333333')

        ax1.bar(
            x + offset,
            perf_means,
            width,
            yerr=perf_stds,
            label=f'{model_label} (performance)',
            color=color,
            alpha=0.7,
            capsize=5,
            error_kw={'linewidth': 2}
        )

    # Customize performance axis
    ax1.set_xlabel('Ablation Configuration', fontsize=16, fontweight='bold')
    ax1.set_ylabel(perf_metric, fontsize=16, fontweight='bold', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([abl.replace('_', '\n') for abl in ablations], rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Create secondary axis for cost
    ax2 = ax1.twinx()

    # Plot cost as lines with markers
    for i, model in enumerate(models):
        model_data = aggregated[aggregated['model'] == model]

        cost_means = []
        cost_stds = []
        for ablation in ablations:
            row = model_data[model_data['ablation'] == ablation]
            if len(row) > 0:
                cost_means.append(row['cost_mean'].values[0])
                cost_stds.append(row['cost_std'].values[0])
            else:
                cost_means.append(0)
                cost_stds.append(0)

        offset = width * (i - 0.5)
        model_label = model.split('/')[-1]
        color = MODEL_COLORS.get(model, '#333333')

        ax2.errorbar(
            x + offset,
            cost_means,
            yerr=cost_stds,
            fmt='o',
            color=color,
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=1.5,
            capsize=5,
            capthick=2,
            elinewidth=2,
            label=f'{model_label} (cost)',
            linestyle='--',
            linewidth=2,
            alpha=0.9
        )

    ax2.set_ylabel('Run cost ($)', fontsize=16, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left',
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_dataset = dataset.replace('_', '-')
    output_file = os.path.join(output_dir, f"bar_chart_{safe_dataset}.pdf")

    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    plt.close()


def plot_bar_charts(output_dir="ablation_res"):
    """Create bar charts for all datasets."""

    print("="*70)
    print("BAR CHART ANALYSIS")
    print("="*70)

    # Fetch data
    df = fetch_runs_data()

    if len(df) == 0:
        print("No data found!")
        return

    # Get unique datasets
    datasets = sorted(df['dataset'].unique())

    print(f"\nDatasets: {datasets}")

    # Create bar chart for each dataset
    for dataset in datasets:
        create_bar_chart(df, dataset, output_dir)

    print("\n" + "="*70)
    print("Done!")


if __name__ == "__main__":
    plot_bar_charts()
