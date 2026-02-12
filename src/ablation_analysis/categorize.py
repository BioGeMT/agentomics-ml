#!/usr/bin/env python3
"""Categorize model architectures using an LLM.

Reads the extracted architectures CSV (which may contain text descriptions or
Python training code) and categorizes each entry into a predefined ML model category.
"""

import os
import time
import pandas as pd
from ollama import Client
from dotenv import load_dotenv
from . import config

CATEGORIES = [
    "Gradient Boosting",
    "Random Forest",
    "Decision Tree",
    "Custom Ensemble",
    "CNN",
    "Linear Regression",
    "Hybrid Model",
    "Transformer",
    "MLP",
    "RNN",
    "SVM",
    "Log. Regression",
    "Foundation Model",
    "Naive Bayes",
]

DEFAULT_MODEL = "gpt-oss:120b"
DEFAULT_DELAY = 3
MAX_ARCHITECTURE_CHARS = 3000


def _init_ollama_client():
    """Initialize Ollama client with credentials from environment."""
    load_dotenv()

    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        raise ValueError("OLLAMA_API_KEY must be set in .env")

    return Client(
        host="https://ollama.com",
        headers={'Authorization': f'Bearer {api_key}'}
    )


def _categorize_single(client, architecture_text, model, delay):
    """Categorize a single architecture entry using the LLM.

    The architecture_text may be a text description or Python code.
    """
    categories_str = '", "'.join(CATEGORIES)

    # Truncate to keep within token limits
    text = architecture_text[:MAX_ARCHITECTURE_CHARS] if len(architecture_text) > MAX_ARCHITECTURE_CHARS else architecture_text

    messages = [
        {
            'role': 'system',
            'content': (
                f'You are categorizing machine learning model architectures. '
                f'The input may be a text description or Python training code. '
                f'Reply with only one of the following categories: "{categories_str}". '
                f'Do not provide any additional text or explanation. '
                f'Reply with exactly one category name.'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'Categorize this into one of the following categories: "{categories_str}".\n\n'
                f'{text}'
            ),
        },
    ]

    time.sleep(delay)

    response = client.chat(model, messages=messages, stream=False)
    category = response['message']['content'].strip()

    # Clean up response
    category = category.strip('"').strip("'").strip()

    return category


def categorize_architectures(input_file, output_file, model=DEFAULT_MODEL, delay=DEFAULT_DELAY):
    """Categorize all architectures in the input CSV.

    Supports resuming: rows that already have a non-empty category value are skipped.
    """
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Run 'extract-arch' first.")
        return None

    print(f"Loading architectures from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Found {len(df)} entries")

    # Resume support: check if output already exists with partial results
    if os.path.isfile(output_file):
        existing = pd.read_csv(output_file)
        if 'category' in existing.columns and len(existing) == len(df):
            already_done = existing['category'].notna().sum()
            if already_done > 0:
                print(f"Resuming: {already_done}/{len(df)} already categorized")
                df['category'] = existing['category']
        else:
            df['category'] = pd.NA
    else:
        df['category'] = pd.NA

    to_categorize = df[df['category'].isna()]
    if to_categorize.empty:
        print("All entries already categorized!")
        _print_summary(df)
        return df

    print(f"Categorizing {len(to_categorize)} entries with model={model}, delay={delay}s\n")

    client = _init_ollama_client()

    for idx in to_categorize.index:
        row = df.loc[idx]
        agent_id = row['agent_id']
        iteration = row['iteration']
        architecture = row['architecture']

        pos = idx + 1
        print(f"[{pos}/{len(df)}] {agent_id} iter {iteration}...", end=' ', flush=True)

        try:
            category = _categorize_single(client, architecture, model, delay)
            df.at[idx, 'category'] = category
            print(f"-> {category}")
        except Exception as e:
            print(f"ERROR: {e}")
            df.at[idx, 'category'] = "Error"

        # Save periodically (every 10 rows) for crash resilience
        if pos % 10 == 0:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            df.to_csv(output_file, index=False)

    # Final save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)

    _print_summary(df)
    return df


def _print_summary(df):
    """Print categorization summary."""
    print(f"\nSaved to categorized CSV")
    print(f"  Total rows: {len(df)}")
    print(f"  Categorized: {df['category'].notna().sum()}")
    print(f"  Errors: {(df['category'] == 'Error').sum()}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().to_string())
