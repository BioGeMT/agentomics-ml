#!/usr/bin/env python3
"""Plot failure mode distributions from failed ablation runs.

Generates a stacked horizontal bar chart: one bar per model,
segments colored by failure category.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config

MODEL_ORDER = [
    'google/gemini-3-flash-preview',
    'x-ai/grok-4.1-fast',
    'gpt-oss:20b',
]

MODEL_LABELS = {
    'google/gemini-3-flash-preview': 'Gemini Flash 3',
    'x-ai/grok-4.1-fast': 'Grok 4.1 Fast',
    'gpt-oss:20b': 'GPT-OSS 20B',
}

CATEGORY_ORDER = [
    'No working output',
    'Faulty inference',
    'Agent corrupted data',
    'LLM API error',
    'LLM output validation failure',
]

CATEGORY_COLORS = {
    'No working output': '#d62728',
    'Faulty inference': '#ff7f0e',
    'Agent corrupted data': '#9467bd',
    'LLM API error': '#1f77b4',
    'LLM output validation failure': '#2ca02c',
}

TICK_SIZE = 25
LABEL_SIZE = 25
LEGEND_SIZE = 22


def plot_failures(input_file, output_dir):
    """Generate stacked horizontal bar chart of failure modes by model."""
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    if 'failure_category' not in df.columns:
        print("Error: 'failure_category' column not found in CSV.")
        return

    print(f"Loaded {len(df)} failed runs from {input_file}")

    models = [m for m in MODEL_ORDER if m in df['model'].unique()]
    categories = [c for c in CATEGORY_ORDER if c in df['failure_category'].unique()]

    # Build count matrix
    data = {}
    for cat in categories:
        cat_df = df[df['failure_category'] == cat]
        model_counts = cat_df['model'].value_counts()
        data[cat] = model_counts.reindex(models, fill_value=0).values

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(right=0.65)
    y = np.arange(len(models))
    bar_height = 0.55
    left = np.zeros(len(models))

    for cat in categories:
        vals = data[cat]
        color = CATEGORY_COLORS.get(cat, 'gray')
        ax.barh(y, vals, bar_height, left=left,
                label=cat, color=color, alpha=0.9,
                edgecolor='white', linewidth=1.0)
        # Annotate counts above each segment
        for j, val in enumerate(vals):
            if val > 0:
                cx = left[j] + val / 2
                ax.text(cx, y[j] + bar_height / 2 + 0.08, str(int(val)),
                        ha='center', va='bottom',
                        fontsize=TICK_SIZE - 6, fontweight='bold', color='#333333')
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=TICK_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.set_xlabel('Number of failed runs', fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.9,
              bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, left.max() * 1.05)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'failures_stacked.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")
