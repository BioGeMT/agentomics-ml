#!/usr/bin/env python3
"""Create bar chart showing cost only for ablation study."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_cost_vs_performance import fetch_runs_data, DATASET_METRICS

# Set style
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 15

# Colors for models
MODEL_COLORS = {
    'google/gemini-3-flash-preview': '#1f77b4',  # blue
    'x-ai/grok-4.1-fast': '#ff7f0e',  # orange
}

MODEL_LABELS = {
    'google/gemini-3-flash-preview': 'Gemini Flash 3',
    'x-ai/grok-4.1-fast': 'Grok 4.1 Fast',
}


def create_cost_bar_chart(df, dataset, output_dir="ablation_res"):
    """Create bar chart showing cost only."""

    # Filter data for this dataset
    data = df[df['dataset'] == dataset].copy()

    if len(data) == 0:
        print(f"Warning: No data for {dataset}")
        return

    # Filter out rows with missing cost
    data = data.dropna(subset=['total_cost'])

    if len(data) == 0:
        print(f"Warning: No valid data for {dataset}")
        return

    models = sorted(data['model'].unique())
    print(f"\nCreating cost bar chart for {dataset}")
    print(f"  Models: {models}")
    print(f"  Runs: {len(data)}")

    # Calculate means and std for cost
    aggregated = data.groupby(['model', 'ablation']).agg({
        'total_cost': ['mean', 'std', 'count']
    }).reset_index()

    aggregated.columns = ['model', 'ablation', 'cost_mean', 'cost_std', 'n']
    aggregated['cost_std'].fillna(0, inplace=True)

    # Get ablations
    ablations = sorted(data['ablation'].unique())
    print(f"  Ablations: {ablations}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Set up bar positions
    x = np.arange(len(ablations))
    width = 0.35

    # Plot cost bars for each model
    for i, model in enumerate(models):
        model_data = aggregated[aggregated['model'] == model]

        cost_means = []
        cost_stds_lower = []
        cost_stds_upper = []

        for ablation in ablations:
            row = model_data[model_data['ablation'] == ablation]
            if len(row) > 0:
                mean = row['cost_mean'].values[0]
                std = row['cost_std'].values[0]

                # Check for negative or zero cost
                if mean <= 0:
                    print(f"  WARNING: {model} × {ablation} has cost = {mean}, skipping")
                    cost_means.append(0)
                    cost_stds_lower.append(0)
                    cost_stds_upper.append(0)
                else:
                    cost_means.append(mean)
                    # Ensure error bar doesn't go below 0
                    cost_stds_lower.append(min(std, mean))
                    cost_stds_upper.append(std)
            else:
                cost_means.append(0)
                cost_stds_lower.append(0)
                cost_stds_upper.append(0)

        # Create asymmetric error bars (don't go below 0)
        yerr = [cost_stds_lower, cost_stds_upper]

        offset = width * (i - 0.5)
        model_label = MODEL_LABELS.get(model, model.split('/')[-1])
        color = MODEL_COLORS.get(model, '#333333')

        ax.bar(
            x + offset,
            cost_means,
            width,
            yerr=yerr,
            label=model_label,
            color=color,
            alpha=0.8,
            capsize=5,
            error_kw={'linewidth': 2.5, 'capthick': 2.5}
        )

    # Customize plot
    ax.set_xlabel('Ablation Configuration', fontsize=18, fontweight='bold')
    ax.set_ylabel('Run cost ($)', fontsize=18, fontweight='bold')
    ax.set_xticks(x)

    # Format x-axis labels (remove "no_", replace underscores, capitalize)
    x_labels = []
    for abl in ablations:
        # Remove "no_" prefix
        label = abl.replace('no_', '').replace('_', ' ')

        # Special case: rename "final outcome" to "Inference"
        if 'final outcome' in label.lower():
            label = 'Inference'
        else:
            # Capitalize first letter of each word
            label = ' '.join(word.capitalize() for word in label.split())

        x_labels.append(label)

    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Reduce margins to bring bars closer to axes
    ax.margins(x=0.02)

    # Legend in top right
    ax.legend(
        loc='upper right',
        fontsize=15,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_dataset = dataset.replace('_', '-')
    output_file = os.path.join(output_dir, f"cost_only_{safe_dataset}.pdf")

    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    plt.close()


def plot_cost_only(output_dir="ablation_res"):
    """Create cost bar charts for all datasets."""

    print("="*70)
    print("COST ANALYSIS (Bar Charts)")
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
        create_cost_bar_chart(df, dataset, output_dir)

    print("\n" + "="*70)
    print("Done!")


if __name__ == "__main__":
    plot_cost_only()
