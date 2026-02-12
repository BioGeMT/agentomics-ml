#!/usr/bin/env python3
"""Plot architecture category distributions from categorized ablation data.

Generates two groups of plots (one per dataset):
1. Comparison: with vs without architecture step
2. By LLM model: distribution per model
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config

# Ordered list — controls bar order (top to bottom) and legend order
MODEL_ORDER = [
    'google/gemini-3-flash-preview',
    'x-ai/grok-4.1-fast',
    'gpt-oss:20b',
]

MODEL_COLORS = {
    'google/gemini-3-flash-preview': '#1f77b4',
    'x-ai/grok-4.1-fast': '#ff7f0e',
    'gpt-oss:20b': '#2ca02c',
}

MODEL_LABELS = {
    'google/gemini-3-flash-preview': 'Gemini Flash 3',
    'x-ai/grok-4.1-fast': 'Grok 4.1 Fast',
    'gpt-oss:20b': 'GPT-OSS 20B',
}

# Ablation configs that skip the architecture step
NO_ARCH_CONFIGS = {'no_model_architecture', 'no_all_steps'}

# Font sizes for dissertation-ready plots
TICK_SIZE = 25
LABEL_SIZE = 25
LEGEND_SIZE = 20


def _get_pct_distribution(subset, all_categories):
    """Compute percentage distribution over categories."""
    counts = subset['category'].value_counts()
    total = len(subset)
    if total == 0:
        return pd.Series(0.0, index=all_categories)
    return (counts.reindex(all_categories, fill_value=0) / total * 100)


def plot_arch_comparison(df, output_dir):
    """Plot with vs without architecture step comparison, merged across datasets."""
    with_arch = df[~df['ablation_config'].isin(NO_ARCH_CONFIGS)]
    without_arch = df[df['ablation_config'].isin(NO_ARCH_CONFIGS)]

    all_categories = sorted(df['category'].unique())
    pct_with = _get_pct_distribution(with_arch, all_categories)
    pct_without = _get_pct_distribution(without_arch, all_categories)

    # Sort by total percentage (ascending so highest is at top)
    total_pct = pct_with + pct_without
    sort_order = total_pct.sort_values().index
    pct_with = pct_with.reindex(sort_order)
    pct_without = pct_without.reindex(sort_order)

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_height = 0.45
    y = np.arange(len(sort_order))

    ax.barh(y + bar_height / 2, pct_with.values, bar_height,
            label=f'With architecture step (n={len(with_arch)})',
            color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.barh(y - bar_height / 2, pct_without.values, bar_height,
            label=f'Without architecture step (n={len(without_arch)})',
            color='coral', alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sort_order, fontsize=TICK_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.set_xlabel('Percentage (%)', fontsize=LABEL_SIZE)
    # No y-axis label — category names are self-explanatory
    ax.legend(fontsize=LEGEND_SIZE, loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'arch_comparison.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_arch_by_model(df, output_dir):
    """Plot architecture distribution per LLM model, one per dataset."""
    datasets = sorted(df['dataset'].unique())
    # Use fixed order, filtered to models actually present in data
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    for dataset in datasets:
        ds = df[df['dataset'] == dataset]

        all_categories = sorted(ds['category'].unique())

        model_pcts = {}
        model_counts = {}
        for model in models:
            subset = ds[ds['model'] == model]
            model_pcts[model] = _get_pct_distribution(subset, all_categories)
            model_counts[model] = len(subset)

        # Sort categories by total percentage across models
        total_pct = sum(model_pcts.values())
        sort_order = total_pct.sort_values().index
        for model in models:
            model_pcts[model] = model_pcts[model].reindex(sort_order)

        fig, ax = plt.subplots(figsize=(10, 7))
        n_models = len(models)
        bar_height = 0.8 / n_models
        y = np.arange(len(sort_order))

        for i, model in enumerate(models):
            # Reverse offset so first model in MODEL_ORDER is the top bar
            offset = ((n_models - 1) / 2 - i) * bar_height
            label = MODEL_LABELS.get(model, model)
            color = MODEL_COLORS.get(model, f'C{i}')
            ax.barh(y + offset, model_pcts[model].values, bar_height,
                    label=f'{label} (n={model_counts[model]})',
                    color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(sort_order, fontsize=TICK_SIZE)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        ax.set_xlabel('Percentage (%)', fontsize=LABEL_SIZE)
        # No y-axis label — category names are self-explanatory
        ax.legend(fontsize=LEGEND_SIZE, loc='lower right', framealpha=0.9)
        ax.grid(axis='x', alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'arch_by_model_{dataset}.pdf')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_architectures(input_file, output_dir):
    """Main entry point: load data and generate all architecture plots."""
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Run 'categorize' first.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} entries from {input_file}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Categories: {sorted(df['category'].unique())}\n")

    os.makedirs(output_dir, exist_ok=True)

    print("--- Architecture step comparison ---")
    plot_arch_comparison(df, output_dir)

    print("\n--- Architecture by model ---")
    plot_arch_by_model(df, output_dir)

    print(f"\nDone. All plots saved to {output_dir}")
